"""Shard assignment for the sharded-vs-centralized vector store comparison.

`num_shards=1` is just the degenerate single-shard case of the same code
path used for `num_shards>1` -- there's no separate "centralized mode"
branch anywhere in this project. Sharding is by `article_id` (not
`paragraph_id`) so every paragraph of a given article lands in the same
shard, keeping article-level locality.
"""

from __future__ import annotations


def assign_shard(article_id: int, num_shards: int) -> int:
    """Deterministic shard assignment: plain `article_id % num_shards`, not
    Spark's internal hash-partitioner, so this can be independently
    recomputed by any future component (e.g. a Part 3 API router) without
    needing to ask Spark how it partitioned anything."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    return article_id % num_shards


def shard_dir_name(shard_id: int) -> str:
    """Directory name for a shard's output, matching Spark's Hive-style
    partitioning convention (`partitionBy("shard_id")` produces `shard_id=N/`
    subdirectories) so Phase B can read them back consistently."""
    return f"shard_id={shard_id}"
