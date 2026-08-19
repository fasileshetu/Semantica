

import numpy as np
import pytest

from embeddings.embedder import Embedder, make_embedder


class FakeEmbedder(Embedder):
    """Deterministic fixed-vector embedder for testing code that depends on
    the Embedder interface without needing a real model."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        # Deterministic: each text maps to a vector derived from its hash,
        # so the same text always embeds to the same vector.
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors[i] = rng.random(self.dim).astype("float32")
        return vectors


def test_fake_embedder_returns_correct_shape() -> None:
    embedder = FakeEmbedder(dim=16)
    vectors = embedder.embed(["hello", "world", "foo"])
    assert vectors.shape == (3, 16)
    assert vectors.dtype == np.float32


def test_fake_embedder_is_deterministic() -> None:
    embedder = FakeEmbedder(dim=8)
    v1 = embedder.embed(["same text"])
    v2 = embedder.embed(["same text"])
    np.testing.assert_array_equal(v1, v2)


def test_fake_embedder_handles_empty_input() -> None:
    embedder = FakeEmbedder(dim=8)
    vectors = embedder.embed([])
    assert vectors.shape == (0, 8)


def test_make_embedder_factory_creates_known_type() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        make_embedder("not-a-real-embedder", model_name="x", device=None)
