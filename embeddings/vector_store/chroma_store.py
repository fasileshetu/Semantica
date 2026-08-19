"""ChromaDB-backed VectorStore.

Unlike FAISS, ChromaDB accepts our string ids (`paragraph_id`) directly --
no int64 mapping problem here. ChromaDB always builds an approximate HNSW
index internally (there's no exact/flat mode in its public API), so a
FAISS-vs-Chroma comparison is not purely a sharding-strategy comparison --
it's also exact-vs-approximate search. Worth remembering when reading
benchmark numbers.

The collection is explicitly configured for cosine distance
(`hnsw:space: cosine`) to match FAISS's inner-product-over-normalized-vectors
convention. Chroma's `query()` returns *distance* (0 = identical, larger =
less similar); we convert to a `score` consistent with FAISS's convention
(higher = more similar) via `score = 1 - distance`, which recovers cosine
similarity when the collection is in cosine space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from embeddings.vector_store.base import SearchResult

COLLECTION_NAME = "semantica"


class ChromaVectorStore:
    def __init__(self, dim: int, path: Path | None = None) -> None:
        self.dim = dim
        self._path = path
        self._client = chromadb.PersistentClient(path=str(path)) if path else chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_batch(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("ids, vectors, and metadata must be the same length")
        if len(ids) == 0:
            return
        # Chroma rejects empty metadata dicts; substitute a harmless placeholder.
        safe_metadata = [md if md else {"_empty": True} for md in metadata]
        self._collection.add(
            ids=ids,
            embeddings=np.ascontiguousarray(vectors, dtype="float32").tolist(),
            metadatas=safe_metadata,
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        query_vector = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        result = self._collection.query(query_embeddings=query_vector.tolist(), n_results=k)

        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]

        return [
            SearchResult(id=id_, score=1.0 - dist, metadata=md or {})
            for id_, dist, md in zip(ids, distances, metadatas, strict=True)
        ]

    def save(self, path: Path) -> None:
        # PersistentClient already writes to disk continuously at `self._path`;
        # `save()` just needs to ensure that's where this store actually lives.
        if self._path is None or Path(self._path) != Path(path):
            raise ValueError(
                "ChromaVectorStore must be constructed with path=<this path> "
                "up front (PersistentClient writes incrementally, not on save())."
            )

    @classmethod
    def load(cls, path: Path) -> ChromaVectorStore:
        # dim isn't recoverable from Chroma's API without peeking at a stored
        # vector; callers that need it should track it separately (e.g. via
        # the embeddings.parquet metadata this store was built from).
        store = cls.__new__(cls)
        store._path = path
        store._client = chromadb.PersistentClient(path=str(path))
        store._collection = store._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        store.dim = None
        return store
