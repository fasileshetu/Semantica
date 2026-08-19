"""Embedding generation, behind an interface so a future external-API-based
embedder can be swapped in later without touching the Spark job.

`generate_embeddings.py`'s Spark stage only ever calls `embedder.embed(texts)`
-- it never needs to know whether that's a local model or an HTTP call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), self.dim) float32 array, one row per input text."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Local model via sentence-transformers. Default `all-MiniLM-L6-v2`
    (384-dim) chosen for speed at this corpus's scale
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        # `get_sentence_embedding_dimension` was renamed to
        # `get_embedding_dimension` in sentence-transformers 5.x; support both
        # so this doesn't break on slightly older installs.
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")


def make_embedder(name: str, model_name: str, device: str | None) -> Embedder:
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=model_name, device=device)
    raise ValueError(f"Unknown embedder: {name!r} (only 'sentence-transformers' is implemented)")
