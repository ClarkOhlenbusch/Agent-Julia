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
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

_ef = None  # lazy — only loads on Brev VM where torch/sentence-transformers are available


def _get_ef():
    global _ef
    if _ef is None:
        import chromadb as _chromadb  # noqa: F401
        from chromadb.utils import embedding_functions as _ef_mod
        _ef = _ef_mod.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return _ef


class MemoryStore:
    def __init__(self) -> None:
        import chromadb
        ef = _get_ef()
        self._client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self._episodic = self._client.get_or_create_collection(
            name="episodic_memory", embedding_function=ef
        )
        self._semantic = self._client.get_or_create_collection(
            name="semantic_memory", embedding_function=ef
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


# ============================================================================
# Module-level shims for app.py / older callers.
# Lazy-initialised singleton MemoryStore.
# ============================================================================

import json as _json
from typing import Optional

_default_store: Optional[MemoryStore] = None


def _store() -> MemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


def episodic_write(text: str, speaker: Optional[str] = None) -> None:
    txt = f"[{speaker}] {text}" if speaker else text
    _store().write_episodic(txt)


def episodic_recent(n: int = 10) -> list[dict]:
    """Return the n most recent episodic entries as dicts (text + ts + speaker?)."""
    s = _store()
    try:
        res = s._episodic.get(include=["metadatas", "documents"])
        rows = list(zip(res["ids"], res["documents"], res["metadatas"]))
        rows.sort(key=lambda r: r[2].get("ts", 0))
        out = []
        for _, doc, meta in rows[-n:]:
            speaker = None
            text = doc
            if doc.startswith("[") and "] " in doc:
                tag, rest = doc.split("] ", 1)
                speaker = tag.lstrip("[")
                text = rest
            out.append({"text": text, "speaker": speaker, "ts": meta.get("ts", 0)})
        return out
    except Exception:
        return []


def episodic_search(query: str, k: int = 3) -> list[dict]:
    docs = _store().query_episodic(query, k=k)
    return [{"text": d, "speaker": None} for d in docs]


def episodic_count() -> int:
    try:
        return _store()._episodic.count()
    except Exception:
        return 0


def semantic_write(facts: list) -> int:
    if not facts:
        return 0
    s = _store()
    written = 0
    for f in facts:
        try:
            subject = getattr(f, "subject", None) or f.get("subject")
            ftype = getattr(f, "type", None) or f.get("type")
            ftype = ftype.value if hasattr(ftype, "value") else ftype
            fact_str = getattr(f, "fact", None) or f.get("fact")
            conf = float(getattr(f, "confidence", None) or f.get("confidence", 0.7))
            s.write_semantic(subject, ftype, fact_str, conf)
            written += 1
        except Exception:
            continue
    return written


def semantic_search(query: str, k: int = 5, min_confidence: float = 0.0) -> list[dict]:
    docs = _store().query_semantic(query, k=k)
    out = []
    for d in docs:
        # query_semantic prepends [subject / type] — re-parse for app display
        text = d
        subject = type_ = None
        if d.startswith("[") and "] " in d:
            tag, body = d.split("] ", 1)
            text = body
            inside = tag.lstrip("[")
            if " / " in inside:
                subject, type_ = [p.strip() for p in inside.split(" / ", 1)]
            else:
                subject = inside
        out.append({"text": text, "subject": subject, "type": type_, "confidence": 1.0})
    return out


def semantic_all() -> list[dict]:
    s = _store()
    try:
        res = s._semantic.get(include=["metadatas", "documents"])
        out = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            out.append({
                "text": doc,
                "subject": meta.get("subject"),
                "type": meta.get("type"),
                "confidence": meta.get("confidence", 0.0),
            })
        out.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return out
    except Exception:
        return []


def semantic_count() -> int:
    try:
        return _store()._semantic.count()
    except Exception:
        return 0


def reset_all() -> None:
    s = _store()
    try:
        s._client.delete_collection("episodic_memory")
        s._client.delete_collection("semantic_memory")
    except Exception:
        pass
    # Re-init the underlying collections
    ef = _get_ef()
    s._episodic = s._client.get_or_create_collection(name="episodic_memory", embedding_function=ef)
    s._semantic = s._client.get_or_create_collection(name="semantic_memory", embedding_function=ef)


def seed_from_file(path: str) -> int:
    try:
        with open(path) as f:
            data = _json.load(f)
    except Exception:
        return 0
    s = _store()
    n = 0
    for item in data or []:
        try:
            s.write_semantic(item["subject"], item["type"], item["fact"],
                             float(item.get("confidence", 0.8)))
            n += 1
        except Exception:
            continue
    return n
