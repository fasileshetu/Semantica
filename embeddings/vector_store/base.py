"""Common vector-store interface, implemented by both backends.

The interface is deliberately string-id-uniform: callers always pass/receive
`paragraph_id`-shaped string ids. FAISS internally needs int64 ids and
handles that mapping itself (see `faiss_store.py`); ChromaDB accepts string
ids natively. Neither backend's id quirks leak through this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """A pluggable vector store backend. `FaissVectorStore` and
    `ChromaVectorStore` both implement this shape; `generate_embeddings.py`
    and `benchmark_query.py` depend only on this interface, never on a
    specific backend's API."""

    def add_batch(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None:
        """Add a batch of (id, vector, metadata) triples. `vectors` is
        (len(ids), dim) float32."""
        ...

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        """Return the top-k nearest results to `query_vector` (shape (dim,)),
        ordered by descending similarity score."""
        ...

    def save(self, path: Path) -> None:
        """Persist the store to `path` (a directory)."""
        ...

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        """Load a previously-`save()`d store back from `path`."""
        ...
