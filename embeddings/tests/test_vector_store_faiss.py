import numpy as np
import pytest

from embeddings.vector_store.faiss_store import FaissVectorStore, hash_id


def _unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_add_and_search_returns_nearest_neighbor() -> None:
    dim = 16
    vecs = _unit_vectors(20, dim)
    ids = [f"p{i}" for i in range(20)]
    metadata = [{"article_id": i // 2} for i in range(20)]

    store = FaissVectorStore(dim=dim)
    store.add_batch(ids, vecs, metadata)

    # Querying with a vector identical to one already in the store should
    # return that vector's id first, with the closest-to-1.0 score.
    results = store.search(vecs[5], k=3)
    assert results[0].id == "p5"
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[0].metadata == {"article_id": 2}
    assert len(results) == 3


def test_search_matches_bruteforce_cosine_similarity() -> None:
    dim = 32
    vecs = _unit_vectors(50, dim, seed=1)
    ids = [f"p{i}" for i in range(50)]
    metadata = [{} for _ in range(50)]

    store = FaissVectorStore(dim=dim)
    store.add_batch(ids, vecs, metadata)

    query = _unit_vectors(1, dim, seed=2)[0]
    results = store.search(query, k=5)

    # Brute-force cosine similarity (vectors are already unit-norm, so dot
    # product == cosine similarity) as the ground truth to compare against.
    sims = vecs @ query
    expected_top5 = [ids[i] for i in np.argsort(-sims)[:5]]

    assert [r.id for r in results] == expected_top5


def test_search_k_greater_than_ntotal_does_not_crash() -> None:
    dim = 8
    vecs = _unit_vectors(3, dim)
    store = FaissVectorStore(dim=dim)
    store.add_batch(["a", "b", "c"], vecs, [{}, {}, {}])

    results = store.search(vecs[0], k=10)
    assert len(results) == 3  # FAISS pads with -1 ids beyond ntotal; those are dropped


def test_save_and_load_round_trip(tmp_path) -> None:
    dim = 12
    vecs = _unit_vectors(10, dim)
    ids = [f"p{i}" for i in range(10)]
    metadata = [{"article_title": f"Article {i}"} for i in range(10)]

    store = FaissVectorStore(dim=dim)
    store.add_batch(ids, vecs, metadata)
    store.save(tmp_path / "faiss_store")

    loaded = FaissVectorStore.load(tmp_path / "faiss_store")
    assert loaded.dim == dim

    results = loaded.search(vecs[3], k=1)
    assert results[0].id == "p3"
    assert results[0].metadata == {"article_title": "Article 3"}


def test_add_batch_rejects_mismatched_lengths() -> None:
    store = FaissVectorStore(dim=4)
    with pytest.raises(ValueError, match="same length"):
        store.add_batch(["a", "b"], _unit_vectors(2, 4), [{}])


def test_add_batch_raises_on_hash_collision(monkeypatch) -> None:
    """Force a collision by making hash_id return a constant, and verify the
    store detects and rejects it rather than silently overwriting."""
    import embeddings.vector_store.faiss_store as faiss_store_module

    monkeypatch.setattr(faiss_store_module, "hash_id", lambda s: 42)

    store = FaissVectorStore(dim=4)
    store.add_batch(["a"], _unit_vectors(1, 4), [{}])
    with pytest.raises(ValueError, match="collision"):
        store.add_batch(["b"], _unit_vectors(1, 4, seed=9), [{}])


def test_hash_id_is_deterministic() -> None:
    assert hash_id("12345_2") == hash_id("12345_2")
    assert hash_id("12345_2") != hash_id("12345_3")
