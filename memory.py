"""
Two-tier memory backed by ChromaDB.

Tier 1 — episodic_memory : raw transcript chunks, 10-min rolling window
Tier 2 — semantic_memory : distilled facts that persist for the session

Both collections use bge-small-en-v1.5 embeddings (CPU, sub-ms writes).
"""
import time
import logging
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_HOST, CHROMA_PORT,
    EPISODIC_TTL_SECONDS, EPISODIC_TOP_K, SEMANTIC_TOP_K,
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)


class MemoryStore:
    def __init__(self) -> None:
        self._client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self._episodic = self._client.get_or_create_collection(
            name="episodic_memory", embedding_function=_ef
        )
        self._semantic = self._client.get_or_create_collection(
            name="semantic_memory", embedding_function=_ef
        )
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
