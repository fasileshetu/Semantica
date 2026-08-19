import pytest

from embeddings.sharding import assign_shard, shard_dir_name


def test_assign_shard_is_deterministic() -> None:
    assert assign_shard(12345, 4) == assign_shard(12345, 4)


def test_assign_shard_stays_in_range() -> None:
    num_shards = 5
    for article_id in range(1000):
        shard = assign_shard(article_id, num_shards)
        assert 0 <= shard < num_shards


def test_assign_shard_single_shard_is_always_zero() -> None:
    for article_id in [0, 1, 42, 999999]:
        assert assign_shard(article_id, 1) == 0


def test_assign_shard_matches_modulo() -> None:
    assert assign_shard(10, 3) == 10 % 3
    assert assign_shard(0, 4) == 0


def test_assign_shard_rejects_invalid_num_shards() -> None:
    with pytest.raises(ValueError, match="num_shards"):
        assign_shard(1, 0)
    with pytest.raises(ValueError, match="num_shards"):
        assign_shard(1, -1)


def test_assign_shard_keeps_articles_together_regardless_of_paragraph_index() -> None:
    # Simulates: all paragraphs of article 777 must land in the same shard.
    article_id = 777
    num_shards = 4
    shards = {assign_shard(article_id, num_shards) for _paragraph_index in range(10)}
    assert len(shards) == 1


def test_shard_dir_name_matches_spark_hive_partitioning_convention() -> None:
    assert shard_dir_name(0) == "shard_id=0"
    assert shard_dir_name(3) == "shard_id=3"
