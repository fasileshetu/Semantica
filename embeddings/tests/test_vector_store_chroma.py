import numpy as np
import pytest

from embeddings.vector_store.chroma_store import ChromaVectorStore


def _unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_add_and_search_returns_nearest_neighbor(tmp_path) -> None:
    dim = 16
    vecs = _unit_vectors(20, dim)
    ids = [f"p{i}" for i in range(20)]
    metadata = [{"article_id": i // 2} for i in range(20)]

    store = ChromaVectorStore(dim=dim, path=tmp_path / "chroma_store")
    store.add_batch(ids, vecs, metadata)

    results = store.search(vecs[5], k=3)
    assert results[0].id == "p5"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[0].metadata["article_id"] == 2
    assert len(results) == 3


def test_search_matches_bruteforce_cosine_similarity_at_small_scale(tmp_path) -> None:
    # Chroma's HNSW index is approximate, but at this small a scale (50
    # points) it should still return the true nearest neighbors.
    dim = 32
    vecs = _unit_vectors(50, dim, seed=1)
    ids = [f"p{i}" for i in range(50)]
    metadata = [{} for _ in range(50)]

    store = ChromaVectorStore(dim=dim, path=tmp_path / "chroma_store")
    store.add_batch(ids, vecs, metadata)

    query = _unit_vectors(1, dim, seed=2)[0]
    results = store.search(query, k=5)

    sims = vecs @ query
    expected_top5 = {ids[i] for i in np.argsort(-sims)[:5]}

    assert {r.id for r in results} == expected_top5


def test_load_and_search_after_reopening_persistent_client(tmp_path) -> None:
    dim = 12
    vecs = _unit_vectors(10, dim)
    ids = [f"p{i}" for i in range(10)]
    metadata = [{"article_title": f"Article {i}"} for i in range(10)]

    path = tmp_path / "chroma_store"
    store = ChromaVectorStore(dim=dim, path=path)
    store.add_batch(ids, vecs, metadata)
    del store  # simulate ending the process

    reloaded = ChromaVectorStore.load(path)
    results = reloaded.search(vecs[3], k=1)
    assert results[0].id == "p3"
    assert results[0].metadata["article_title"] == "Article 3"


def test_add_batch_rejects_mismatched_lengths(tmp_path) -> None:
    store = ChromaVectorStore(dim=4, path=tmp_path / "chroma_store")
    with pytest.raises(ValueError, match="same length"):
        store.add_batch(["a", "b"], _unit_vectors(2, 4), [{}])


def test_add_batch_handles_empty_metadata(tmp_path) -> None:
    """Chroma rejects genuinely-empty metadata dicts -- confirm the store's
    placeholder substitution keeps add_batch working when callers pass {}."""
    store = ChromaVectorStore(dim=4, path=tmp_path / "chroma_store")
    store.add_batch(["a"], _unit_vectors(1, 4), [{}])
    results = store.search(_unit_vectors(1, 4)[0], k=1)
    assert results[0].id == "a"
