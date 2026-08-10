"""
Semantica API — search layer scaffold.

Stub implementation using fake in-memory data so the API shape can be built
and tested before real embeddings/vector store are wired in. Swap out
STUB_DOCS and the embed/search logic once Archit's pipeline produces real
vectors.
"""

import random
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(
    title="Semantica API",
    description="Semantic search over Wikipedia paragraphs.",
    version="0.1.0",
)

# --- Stub data (replace with real vector store once available) ---

STUB_DOCS = [
    {"doc_id": "1", "title": "OK!", "text": "OK! is a British weekly magazine specialising in royal and celebrity news."},
    {"doc_id": "2", "title": "Spark (software)", "text": "Apache Spark is a unified analytics engine for large-scale data processing."},
    {"doc_id": "3", "title": "Wikipedia", "text": "Wikipedia is a free online encyclopedia edited collaboratively."},
    {"doc_id": "4", "title": "Vector database", "text": "A vector database stores data as high-dimensional vectors for similarity search."},
    {"doc_id": "5", "title": "REST API", "text": "REST is an architectural style for designing networked applications."},
]


def fake_embed(text: str) -> list[float]:
    """Stand-in for a real embedding model. Deterministic-ish fake vector."""
    random.seed(hash(text) % (2**32))
    return [round(random.uniform(-1, 1), 4) for _ in range(8)]


def fake_similarity_score() -> float:
    return round(random.uniform(0.5, 0.99), 4)


# --- Schemas ---

class SearchResult(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    took_ms: float


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    text: str
    embedding: list[float]
    dims: int


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Max results to return"),
):
    """
    Semantic search over indexed Wikipedia paragraphs.
    Currently returns stub results with fake similarity scores.
    """
    start = time.perf_counter()

    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Stub: pretend every doc matches with a random score, sorted descending
    results = [
        SearchResult(
            doc_id=doc["doc_id"],
            title=doc["title"],
            text=doc["text"],
            score=fake_similarity_score(),
        )
        for doc in STUB_DOCS[:limit]
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    took_ms = round((time.perf_counter() - start) * 1000, 2)
    return SearchResponse(query=q, results=results, took_ms=took_ms)


@app.get("/similar/{doc_id}", response_model=SearchResponse)
def similar(doc_id: str, limit: int = Query(5, ge=1, le=50)):
    """
    Find paragraphs semantically similar to a given document.
    Currently returns stub results.
    """
    source = next((d for d in STUB_DOCS if d["doc_id"] == doc_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' not found")

    start = time.perf_counter()
    others = [d for d in STUB_DOCS if d["doc_id"] != doc_id][:limit]
    results = [
        SearchResult(
            doc_id=doc["doc_id"],
            title=doc["title"],
            text=doc["text"],
            score=fake_similarity_score(),
        )
        for doc in others
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    took_ms = round((time.perf_counter() - start) * 1000, 2)
    return SearchResponse(query=f"similar to {doc_id}", results=results, took_ms=took_ms)


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """
    Generate an embedding for arbitrary text.
    Currently returns a fake deterministic vector.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    vector = fake_embed(request.text)
    return EmbedResponse(text=request.text, embedding=vector, dims=len(vector))