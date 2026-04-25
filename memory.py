"""Two-tier memory.

Tier 1 — episodic: in-memory rolling buffer (~10 min). No persistence needed.
Tier 2 — semantic: distilled facts in ChromaDB. Written when triage says STORE.

Embedded with bge-small-en-v1.5 (loaded once, kept in process).
ChromaDB server runs on host port 8001.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from schema import Fact

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
EPISODIC_TTL_S = 600.0  # 10 minutes

_client: Optional[chromadb.HttpClient] = None
_embedder: Optional[SentenceTransformer] = None
_semantic = None

# In-memory rolling buffer for episodic conversation history
_episodic_buffer: deque[dict] = deque()


def _ensure():
    global _client, _embedder, _semantic
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if _semantic is None:
        _semantic = _client.get_or_create_collection("semantic_memory")


def _embed(text: str) -> list[float]:
    _ensure()
    return _embedder.encode([text], normalize_embeddings=True)[0].tolist()


def _purge_expired():
    cutoff = time.time() - EPISODIC_TTL_S
    while _episodic_buffer and _episodic_buffer[0]["ts"] < cutoff:
        _episodic_buffer.popleft()


# ============================================================================
# Episodic — in-memory rolling buffer
# ============================================================================

def episodic_write(text: str, speaker: Optional[str] = None) -> None:
    _purge_expired()
    _episodic_buffer.append({
        "text": text,
        "speaker": speaker or "unknown",
        "ts": time.time(),
    })


def episodic_recent(n: int = 10) -> list[dict]:
    _purge_expired()
    items = list(_episodic_buffer)[-n:]
    return [{"text": c["text"], "speaker": c["speaker"], "ts": c["ts"]} for c in items]


def episodic_search(query: str, k: int = 3, max_age_s: float = EPISODIC_TTL_S) -> list[dict]:
    """Simple recency-biased keyword search over the buffer."""
    _purge_expired()
    if not _episodic_buffer:
        return []
    now = time.time()
    # Embed query and score against buffer entries
    q_emb = _embed(query)
    import numpy as np
    scored = []
    for entry in _episodic_buffer:
        if now - entry["ts"] > max_age_s:
            continue
        e_emb = _embed(entry["text"])
        sim = float(np.dot(q_emb, e_emb))
        scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"text": e["text"], "speaker": e["speaker"], "ts": e["ts"],
         "age_s": round(now - e["ts"], 1)}
        for _, e in scored[:k]
    ]


def episodic_count() -> int:
    _purge_expired()
    return len(_episodic_buffer)


# ============================================================================
# Semantic — distilled facts in ChromaDB
# ============================================================================

def semantic_write(facts: list[Fact]) -> int:
    if not facts:
        return 0
    _ensure()
    docs, metas, ids, embs = [], [], [], []
    for i, f in enumerate(facts):
        text = f"{f.subject} — {f.fact}"
        docs.append(text)
        metas.append({
            "subject": f.subject,
            "type": f.type.value,
            "confidence": float(f.confidence),
            "key": f"{f.subject}|{f.fact}",
        })
        ids.append(f"sem_{int(time.time()*1000)}_{i}")
        embs.append(_embed(text))
    _semantic.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return len(ids)


def semantic_search(query: str, k: int = 5, min_confidence: float = 0.5) -> list[dict]:
    _ensure()
    if _semantic.count() == 0:
        return []
    res = _semantic.query(
        query_embeddings=[_embed(query)],
        n_results=min(k * 2, _semantic.count()),
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta.get("confidence", 0) < min_confidence:
            continue
        out.append({
            "text": doc,
            "subject": meta.get("subject"),
            "type": meta.get("type"),
            "confidence": meta.get("confidence"),
            "distance": round(dist, 3),
        })
        if len(out) >= k:
            break
    return out


def semantic_all() -> list[dict]:
    _ensure()
    if _semantic.count() == 0:
        return []
    data = _semantic.get()
    out = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        out.append({
            "text": doc,
            "subject": meta.get("subject"),
            "type": meta.get("type"),
            "confidence": meta.get("confidence"),
        })
    out.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return out


def semantic_count() -> int:
    _ensure()
    return _semantic.count()


def seed_from_file(path: str) -> int:
    with open(path) as f:
        data = json.load(f)
    facts = [Fact(**item) for item in data]
    return semantic_write(facts)


# ============================================================================
# Reset
# ============================================================================

def reset_all() -> None:
    global _semantic
    _episodic_buffer.clear()
    _ensure()
    _client.delete_collection("semantic_memory")
    _semantic = _client.get_or_create_collection("semantic_memory")


if __name__ == "__main__":
    print(f"Connecting to chroma at {CHROMA_HOST}:{CHROMA_PORT} ...")
    print(f"Episodic count: {episodic_count()}")
    print(f"Semantic count: {semantic_count()}")
    episodic_write("Yo, want to grab drinks tonight?", speaker="alex")
    print(f"Episodic count after write: {episodic_count()}")
    print("Recent:", episodic_recent(5))
    seeded = seed_from_file(os.path.join(os.path.dirname(__file__), "data/seed_facts.json"))
    print(f"Seeded {seeded} semantic facts")
    print("Semantic search:", semantic_search("when does Julie like to meet"))
