"""FAISS-backed VectorStore: exact (flat) cosine search via IndexFlatIP.

FAISS's `IndexIDMap` requires int64 ids, but our ids are strings like
`"12345_2"` (paragraph_id). We derive a stable int64 id via a hash rather
than a sequential counter, so the mapping is reproducible across separate
runs / partial re-embeds with no cross-partition coordination needed (see
module docstring rationale in the plan -- collision probability at ~1.5M
items over a 64-bit space is negligible, but `add_batch` still asserts no
collisions within what it's tracked, rather than silently overwriting).

A side `id_map.parquet` (written by `save()`, read by `load()`) maps the
int64 FAISS id back to the original string id + arbitrary metadata dict,
since FAISS itself only ever returns ids and scores, never metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embeddings.vector_store.base import SearchResult

ID_MAP_SCHEMA = pa.schema(
    [
        pa.field("vector_id", pa.int64()),
        pa.field("paragraph_id", pa.string()),
        pa.field("metadata_json", pa.string()),
    ]
)


def hash_id(string_id: str) -> int:
    """Deterministically map a string id to an int64 FAISS id."""
    digest = hashlib.blake2b(string_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class FaissVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._id_map: dict[int, tuple[str, dict[str, Any]]] = {}

    def add_batch(
        self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError("ids, vectors, and metadata must be the same length")
        if len(ids) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss_ids = np.array([hash_id(i) for i in ids], dtype="int64")

        for fid, sid in zip(faiss_ids, ids, strict=True):
            existing = self._id_map.get(int(fid))
            if existing is not None and existing[0] != sid:
                raise ValueError(
                    f"FAISS id collision: {sid!r} hashes to the same int64 id as "
                    f"already-added {existing[0]!r}"
                )

        self._index.add_with_ids(vectors, faiss_ids)
        for fid, sid, md in zip(faiss_ids, ids, metadata, strict=True):
            self._id_map[int(fid)] = (sid, md)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchResult]:
        query_vector = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        scores, faiss_ids = self._index.search(query_vector, k)

        results = []
        for score, fid in zip(scores[0], faiss_ids[0], strict=True):
            if fid == -1:
                continue  # FAISS pads with -1 when fewer than k results exist
            paragraph_id, metadata = self._id_map[int(fid)]
            results.append(SearchResult(id=paragraph_id, score=float(score), metadata=metadata))
        return results

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))

        rows = [
            {
                "vector_id": fid,
                "paragraph_id": sid,
                "metadata_json": json.dumps(md, default=str),
            }
            for fid, (sid, md) in self._id_map.items()
        ]
        table = pa.Table.from_pylist(rows, schema=ID_MAP_SCHEMA)
        pq.write_table(table, path / "id_map.parquet")

    @classmethod
    def load(cls, path: Path) -> FaissVectorStore:
        index = faiss.read_index(str(path / "index.faiss"))
        store = cls(dim=index.d)
        store._index = index

        table = pq.read_table(path / "id_map.parquet")
        for row in table.to_pylist():
            store._id_map[row["vector_id"]] = (
                row["paragraph_id"],
                json.loads(row["metadata_json"]),
            )
        return store
