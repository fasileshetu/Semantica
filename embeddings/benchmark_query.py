"""Centralized-vs-sharded query latency benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import typer

from benchmarks.bench_common.logging_utils import setup_logging
from benchmarks.bench_common.metrics import Timer, record_metric
from embeddings.vector_store.base import SearchResult
from embeddings.vector_store.chroma_store import ChromaVectorStore
from embeddings.vector_store.faiss_store import FaissVectorStore

log = setup_logging("benchmark_query")
app = typer.Typer(add_completion=False)

RESULTS_PATH = Path("benchmarks/results/part2_metrics.jsonl")


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _load_store(backend: str, shard_dir: Path):
    if backend == "faiss":
        return FaissVectorStore.load(shard_dir)
    if backend == "chroma":
        return ChromaVectorStore.load(shard_dir)
    raise ValueError(f"Unknown backend: {backend!r}")


def _discover_shards(vectors_dir: Path, backend: str) -> list[Path]:
    backend_dir = vectors_dir / backend
    return sorted(backend_dir.glob("shard_*"), key=lambda p: p.name)


def _merge_top_k(results_per_shard: list[list[SearchResult]], k: int) -> list[SearchResult]:
    all_results = [r for shard_results in results_per_shard for r in shard_results]
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:k]


@app.command()
def main(
    vectors_dir: Path = typer.Option(
        Path("data/vectors/"), help="Output dir from generate_embeddings.py."
    ),
    backend: str = typer.Option("faiss", help="'faiss' or 'chroma'."),
    k: int = typer.Option(10, help="Top-k results to retrieve per query."),
) -> None:
    """Benchmark query latency: centralized (1 shard) vs sharded (>1 shard) index."""
    queries_path = vectors_dir / "sample_queries.parquet"
    if not queries_path.exists():
        raise typer.BadParameter(f"{queries_path} not found -- run generate_embeddings.py first.")

    queries = pq.read_table(queries_path).to_pylist()
    shard_dirs = _discover_shards(vectors_dir, backend)
    if not shard_dirs:
        raise typer.BadParameter(f"No shards found under {vectors_dir / backend}")

    log.info(
        "loading stores",
        extra={"backend": backend, "num_shards": len(shard_dirs), "num_queries": len(queries)},
    )
    stores = [_load_store(backend, d) for d in shard_dirs]
    index_size_bytes = sum(_dir_size_bytes(d) for d in shard_dirs)

    is_sharded = len(stores) > 1
    stage = "query_sharded" if is_sharded else "query_centralized"

    durations = []
    for query in queries:
        query_vector = np.array(query["embedding"], dtype="float32")
        with Timer() as t:
            if is_sharded:
                per_shard_results = [store.search(query_vector, k) for store in stores]
                _merge_top_k(per_shard_results, k)
            else:
                stores[0].search(query_vector, k)
        durations.append(t.elapsed_s or 0.0)

    durations_sorted = sorted(durations)
    p50 = durations_sorted[len(durations_sorted) // 2]
    p95 = durations_sorted[int(len(durations_sorted) * 0.95)]
    mean = sum(durations) / len(durations)

    record_metric(
        stage,
        RESULTS_PATH,
        duration_s=mean,
        extra={
            "backend": backend,
            "num_shards": len(stores),
            "k": k,
            "num_queries": len(queries),
            "p50_s": p50,
            "p95_s": p95,
            "index_size_bytes": index_size_bytes,
        },
    )
    log.info(
        f"{stage} complete",
        extra={
            "backend": backend,
            "num_shards": len(stores),
            "mean_s": mean,
            "p50_s": p50,
            "p95_s": p95,
            "index_size_bytes": index_size_bytes,
        },
    )
    typer.echo(
        f"[{backend}, {len(stores)} shard(s)] mean={mean * 1000:.2f}ms p50={p50 * 1000:.2f}ms "
        f"p95={p95 * 1000:.2f}ms over {len(queries)} queries, index_size={index_size_bytes:,} bytes"
    )


if __name__ == "__main__":
    app()
