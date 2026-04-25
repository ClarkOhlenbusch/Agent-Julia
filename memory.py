"""
Two-tier memory backed by ChromaDB.

Tier 1 — episodic_memory : raw transcript chunks, 10-min rolling window
Tier 2 — semantic_memory : distilled facts that persist for the session

Both collections use bge-small-en-v1.5 embeddings (CPU, sub-ms writes).
"""
import time
import logging
from typing import List

from config import (
    CHROMA_HOST, CHROMA_PORT,
    EPISODIC_TTL_SECONDS, EPISODIC_TOP_K, SEMANTIC_TOP_K,
)

log = logging.getLogger(__name__)

_ef = None  # lazy — uses ONNX MiniLM (no torch/numpy required on client)


def _get_ef():
    global _ef
    if _ef is None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        _ef = ONNXMiniLM_L6_V2()
    return _ef


class _InMemoryFallback:
    """Simple list-based fallback used when ChromaDB is unreachable."""
    def __init__(self):
        self._docs: list[tuple[str, float]] = []  # (text, ts)

    def add(self, ids, documents, metadatas=None):
        for doc, meta in zip(documents, metadatas or [{}] * len(documents)):
            self._docs.append((doc, meta.get("ts", time.time())))

    def get(self, include=None):
        ids = [str(i) for i in range(len(self._docs))]
        metas = [{"ts": ts} for _, ts in self._docs]
        return {"ids": ids, "metadatas": metas}

    def delete(self, ids):
        keep = [i for i in range(len(self._docs)) if str(i) not in ids]
        self._docs = [self._docs[i] for i in keep]

    def count(self):
        return len(self._docs)

    def query(self, query_texts, n_results=3):
        docs = [d for d, _ in self._docs[-n_results:]]
        return {"documents": [docs], "metadatas": [[{}] * len(docs)]}


class MemoryStore:
    def __init__(self) -> None:
        try:
            import chromadb
            ef = _get_ef()
            self._client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self._episodic = self._client.get_or_create_collection(
                name="episodic_memory", embedding_function=ef
            )
            self._semantic = self._client.get_or_create_collection(
                name="semantic_memory", embedding_function=ef
            )
            log.info("MemoryStore connected to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
        except Exception as exc:
            log.warning("ChromaDB unavailable (%s) — using in-memory fallback", exc)
            self._client  = None
            self._episodic = _InMemoryFallback()
            self._semantic = _InMemoryFallback()
        self._ep_counter = 0

    # ------------------------------------------------------------------
    # Episodic
    # ------------------------------------------------------------------

    def write_episodic(self, text: str) -> None:
        if not text.strip():
            return
        self._ep_counter += 1
        doc_id = f"ep_{int(time.time())}_{self._ep_counter}"
        self._episodic.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[{"ts": time.time()}],
        )
        self._evict_episodic()

    def query_episodic(self, query: str, k: int = EPISODIC_TOP_K) -> List[str]:
        try:
            res = self._episodic.query(query_texts=[query], n_results=min(k, self._episodic.count()))
            return res["documents"][0] if res["documents"] else []
        except Exception:
            return []

    def _evict_episodic(self) -> None:
        cutoff = time.time() - EPISODIC_TTL_SECONDS
        try:
            res = self._episodic.get(include=["metadatas"])
            old_ids = [
                doc_id
                for doc_id, meta in zip(res["ids"], res["metadatas"])
                if meta.get("ts", 0) < cutoff
            ]
            if old_ids:
                self._episodic.delete(ids=old_ids)
                log.debug("Evicted %d old episodic chunks", len(old_ids))
        except Exception as exc:
            log.debug("Eviction error: %s", exc)

    # ------------------------------------------------------------------
    # Semantic
    # ------------------------------------------------------------------

    def write_semantic(self, subject: str, fact_type: str, fact: str, confidence: float) -> None:
        doc_id = f"sem_{subject}_{fact_type}_{int(time.time())}"
        self._semantic.add(
            ids=[doc_id],
            documents=[fact],
            metadatas=[{"subject": subject, "type": fact_type, "confidence": confidence, "ts": time.time()}],
        )

    def query_semantic(self, query: str, k: int = SEMANTIC_TOP_K) -> List[str]:
        try:
            count = self._semantic.count()
            if count == 0:
                return []
            res = self._semantic.query(query_texts=[query], n_results=min(k, count))
            docs  = res["documents"][0] if res["documents"] else []
            metas = res["metadatas"][0] if res["metadatas"] else []
            return [
                f"[{m.get('subject','?')} / {m.get('type','?')}] {d}"
                for d, m in zip(docs, metas)
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Combined context string for LLM prompts
    # ------------------------------------------------------------------

    def context_for(self, query: str) -> str:
        ep   = self.query_episodic(query)
        sem  = self.query_semantic(query)
        parts = []
        if ep:
            parts.append("Recent conversation:\n" + "\n".join(f"- {t}" for t in ep))
        if sem:
            parts.append("Known facts:\n" + "\n".join(f"- {f}" for f in sem))
        return "\n\n".join(parts)
