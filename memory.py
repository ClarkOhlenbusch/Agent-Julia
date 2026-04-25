"""Two-tier memory backed by ChromaDB.

Tier 1 — episodic_memory: rolling 10-min transcript chunks.
Tier 2 — semantic_memory: distilled facts (preferences/relationships/decisions).

Embedded with bge-small-en-v1.5 (loaded once, kept in process).
Server runs on host port 8001; from inside NemoClaw, reach it at
host.openshell.internal:8001.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from schema import Fact

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
EPISODIC_TTL_S = 600.0  # 10 minutes

_client: Optional[chromadb.HttpClient] = None
_embedder: Optional[SentenceTransformer] = None
_episodic = None
_semantic = None


def _ensure() -> tuple[chromadb.HttpClient, SentenceTransformer]:
    global _client, _embedder, _episodic, _semantic
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if _episodic is None:
        _episodic = _client.get_or_create_collection("episodic_memory")
    if _semantic is None:
        _semantic = _client.get_or_create_collection("semantic_memory")
    return _client, _embedder


def _embed(text: str) -> list[float]:
    _, embedder = _ensure()
    return embedder.encode([text], normalize_embeddings=True)[0].tolist()


# ============================================================================
# Episodic — every transcript chunk
# ============================================================================

def episodic_write(text: str, speaker: Optional[str] = None) -> str:
    _ensure()
    now = time.time()
    chunk_id = f"ep_{now:.4f}"
    _episodic.add(
        ids=[chunk_id],
        embeddings=[_embed(text)],
        documents=[text],
        metadatas=[{"speaker": speaker or "unknown", "ts": now}],
    )
    return chunk_id


def episodic_search(query: str, k: int = 3, max_age_s: float = EPISODIC_TTL_S) -> list[dict]:
    _ensure()
    if _episodic.count() == 0:
        return []
    now = time.time()
    res = _episodic.query(
        query_embeddings=[_embed(query)],
        n_results=min(k * 3, _episodic.count()),
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if now - meta["ts"] > max_age_s:
            continue
        out.append(
            {
                "text": doc,
                "speaker": meta.get("speaker"),
                "ts": meta["ts"],
                "age_s": round(now - meta["ts"], 1),
                "distance": round(dist, 3),
            }
        )
        if len(out) >= k:
            break
    return out


def episodic_recent(n: int = 10) -> list[dict]:
    """Return the n most recent episodic chunks in chronological order."""
    _ensure()
    if _episodic.count() == 0:
        return []
    data = _episodic.get()
    pairs = list(zip(data["ids"], data["documents"], data["metadatas"]))
    pairs.sort(key=lambda p: p[2]["ts"], reverse=True)
    out = []
    for _, doc, meta in pairs[:n]:
        out.append({"text": doc, "speaker": meta.get("speaker"), "ts": meta["ts"]})
    return list(reversed(out))


def episodic_purge_old(max_age_s: float = EPISODIC_TTL_S) -> int:
    _ensure()
    if _episodic.count() == 0:
        return 0
    cutoff = time.time() - max_age_s
    data = _episodic.get()
    to_del = [i for i, m in zip(data["ids"], data["metadatas"]) if m["ts"] < cutoff]
    if to_del:
        _episodic.delete(ids=to_del)
    return len(to_del)


# ============================================================================
# Semantic — distilled facts
# ============================================================================

def semantic_write(facts: list[Fact]) -> int:
    if not facts:
        return 0
    _ensure()
    docs, metas, ids, embs = [], [], [], []
    for i, f in enumerate(facts):
        text = f"{f.subject} — {f.fact}"
        docs.append(text)
        metas.append(
            {
                "subject": f.subject,
                "type": f.type.value,
                "confidence": float(f.confidence),
                "key": f"{f.subject}|{f.fact}",
            }
        )
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
        out.append(
            {
                "text": doc,
                "subject": meta.get("subject"),
                "type": meta.get("type"),
                "confidence": meta.get("confidence"),
                "distance": round(dist, 3),
            }
        )
        if len(out) >= k:
            break
    return out


def semantic_all() -> list[dict]:
    """Full dump for the UI 'what I know' panel."""
    _ensure()
    if _semantic.count() == 0:
        return []
    data = _semantic.get()
    out = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        out.append(
            {
                "text": doc,
                "subject": meta.get("subject"),
                "type": meta.get("type"),
                "confidence": meta.get("confidence"),
            }
        )
    out.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return out


# ============================================================================
# Counts + seeding
# ============================================================================

def episodic_count() -> int:
    _ensure()
    return _episodic.count()


def semantic_count() -> int:
    _ensure()
    return _semantic.count()


def seed_from_file(path: str) -> int:
    with open(path) as f:
        data = json.load(f)
    facts = [Fact(**item) for item in data]
    return semantic_write(facts)


def reset_all() -> None:
    """Wipe both collections — useful between rehearsals."""
    global _episodic, _semantic
    _ensure()
    _client.delete_collection("episodic_memory")
    _client.delete_collection("semantic_memory")
    _episodic = _client.get_or_create_collection("episodic_memory")
    _semantic = _client.get_or_create_collection("semantic_memory")


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print(f"Connecting to chroma at {CHROMA_HOST}:{CHROMA_PORT} ...")
    print(f"Episodic count: {episodic_count()}")
    print(f"Semantic count: {semantic_count()}")
    cid = episodic_write("Yo, want to grab drinks tonight?", speaker="alex")
    print(f"Wrote episodic chunk {cid}")
    print("Search 'meeting time':", episodic_search("meeting time"))
    seeded = seed_from_file(os.path.join(os.path.dirname(__file__), "data/seed_facts.json"))
    print(f"Seeded {seeded} semantic facts")
    print("Semantic search 'when does Julie like to meet':",
          semantic_search("when does Julie like to meet"))
