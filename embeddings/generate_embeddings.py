"""Part 2 main CLI: read cleaned paragraphs, compute embeddings via Spark,
build FAISS/Chroma vector store index(es).

Two-phase design

- Phase A (Spark, parallel, the benchmarkable part): reads
  `data/clean/paragraphs.parquet`, assigns each row a `shard_id`, computes
  embeddings via `mapInPandas` (model loaded once per partition, then
  batch-encoded -- not per-row), writes `(paragraph_id, article_id,
  article_title, shard_id, embedding)` rows out as Hive-partitioned Parquet.
- Phase B (single-process, driver, after Spark stage): FAISS/Chroma index
  construction is inherently single-process -- you can't have N Spark
  executors concurrently mutate one shared index. So this reads each
  shard's embeddings back and builds/saves its index(es) directly.

Run via `uv run python embeddings/generate_embeddings.py --workers N ...`
(consistent with every other script in this project)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from benchmarks.bench_common.logging_utils import setup_logging
from benchmarks.bench_common.metrics import Timer, record_metric
from embeddings.embedder import DEFAULT_MODEL_NAME, make_embedder
from embeddings.vector_store.chroma_store import ChromaVectorStore
from embeddings.vector_store.faiss_store import FaissVectorStore

log = setup_logging("generate_embeddings")
app = typer.Typer(add_completion=False)

RESULTS_PATH = Path("benchmarks/results/part2_metrics.jsonl")
PHASE_B_BATCH_SIZE = 5000  # bound peak memory when adding vectors to a store

EMBED_OUTPUT_SCHEMA = StructType(
    [
        StructField("paragraph_id", StringType()),
        StructField("article_id", LongType()),
        StructField("article_title", StringType()),
        StructField("shard_id", IntegerType()),
        StructField("embedding", ArrayType(FloatType())),
    ]
)


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# --------------------------------------------------------------------------
# Phase A: Spark
# --------------------------------------------------------------------------


def _make_embed_partition_fn(embedder_name: str, model_name: str, device: str | None):
    """Returns the function passed to `mapInPandas`. Constructs the Embedder
    once per partition (captured in this closure's first iteration), not
    once per row -- that's what amortizes model-load cost."""

    def embed_partition(iterator):
        embedder = None
        for batch in iterator:
            if len(batch) == 0:
                continue
            if embedder is None:
                embedder = make_embedder(embedder_name, model_name, device)
            vectors = embedder.embed(batch["text"].tolist())
            yield pd.DataFrame(
                {
                    "paragraph_id": batch["paragraph_id"].astype(str).tolist(),
                    "article_id": batch["article_id"].tolist(),
                    "article_title": batch["article_title"].tolist(),
                    "shard_id": batch["shard_id"].tolist(),
                    "embedding": [v.tolist() for v in vectors],
                }
            )

    return embed_partition


def run_embedding_stage(
    spark: SparkSession,
    input_path: Path,
    embeddings_dir: Path,
    *,
    workers: int,
    num_shards: int,
    limit: int,
    embedder_name: str,
    model_name: str,
    device: str | None,
) -> int:
    """Phase A. Returns the number of paragraphs processed."""
    df: DataFrame = spark.read.parquet(str(input_path))
    if limit > 0:
        df = df.limit(limit)

    num_paragraphs = df.count()  # cheap: Parquet row counts come from file footers

    df = df.withColumn("shard_id", F.pmod(F.col("article_id"), F.lit(num_shards)).cast("int"))

    # Repartition for real parallelism -- if the source read yields fewer
    # partitions than `workers`, local[N] scaling has nothing to split.
    target_partitions = max(workers * 4, df.rdd.getNumPartitions())
    df = df.repartition(target_partitions)

    result_df = df.select(
        "paragraph_id", "article_id", "article_title", "shard_id", "text"
    ).mapInPandas(
        _make_embed_partition_fn(embedder_name, model_name, device),
        schema=EMBED_OUTPUT_SCHEMA,
    )

    result_df.write.mode("overwrite").partitionBy("shard_id").parquet(str(embeddings_dir))
    return num_paragraphs


# --------------------------------------------------------------------------
# Sample queries: precomputed so benchmark_query.py never needs to import
# sentence-transformers itself (see module docstring note on the FAISS/torch
# OpenMP conflict below in Phase B).
# --------------------------------------------------------------------------


def write_sample_queries(embeddings_dir: Path, output_path: Path, num_samples: int = 10) -> int:
    """Carve off a handful of already-computed embeddings to use as
    benchmark_query.py's query vectors -- "find nearest neighbors to this
    existing paragraph" is a standard, valid latency-benchmark query shape,
    and reusing already-computed vectors costs nothing extra. Returns the
    number of samples actually written."""
    shard_dirs = sorted(embeddings_dir.glob("shard_id=*"))
    if not shard_dirs:
        return 0

    table = pq.read_table(str(shard_dirs[0])).slice(0, num_samples)
    pq.write_table(
        table.select(["paragraph_id", "embedding"]), output_path / "sample_queries.parquet"
    )
    return table.num_rows


# --------------------------------------------------------------------------
# Phase B: single-process index build
#
# IMPORTANT: this process must never import `sentence_transformers`/`torch`
# in the same process as `faiss` -- doing so causes a hard SIGSEGV crash on
# this platform (both link conflicting OpenMP runtimes; `KMP_DUPLICATE_LIB_OK`
# and import-order workarounds were tried and do not reliably fix it, they
# just move where the crash happens). `embedder.py` lazily imports
# sentence-transformers only inside `SentenceTransformerEmbedder.__init__`,
# which is only ever called from within Spark executor subprocesses (see
# `_make_embed_partition_fn` above) -- never from this driver process. Keep
# it that way: this driver process only ever loads precomputed embeddings
# from Parquet (see `write_sample_queries` above and `benchmark_query.py`).
# --------------------------------------------------------------------------


def build_indices_for_shard(
    embeddings_dir: Path,
    output_path: Path,
    shard_id: int,
    *,
    backends: list[str],
) -> dict:
    """Phase B for one shard. Returns per-backend stats for logging."""
    shard_path = embeddings_dir / f"shard_id={shard_id}"
    if not shard_path.exists():
        log.warning("shard has no data, skipping", extra={"shard_id": shard_id})
        return {}

    table = pq.read_table(str(shard_path))
    ids = table["paragraph_id"].to_pylist()
    article_ids = table["article_id"].to_pylist()
    article_titles = table["article_title"].to_pylist()
    embeddings = np.array(table["embedding"].to_pylist(), dtype="float32")
    metadata = [
        {"article_id": aid, "article_title": title}
        for aid, title in zip(article_ids, article_titles, strict=True)
    ]
    dim = embeddings.shape[1]

    stats = {}
    for backend in backends:
        with Timer() as t:
            if backend == "faiss":
                store = FaissVectorStore(dim=dim)
            elif backend == "chroma":
                store_path = output_path / "chroma" / f"shard_{shard_id}"
                store = ChromaVectorStore(dim=dim, path=store_path)
            else:
                raise ValueError(f"Unknown backend: {backend!r}")

            for i in range(0, len(ids), PHASE_B_BATCH_SIZE):
                store.add_batch(
                    ids[i : i + PHASE_B_BATCH_SIZE],
                    embeddings[i : i + PHASE_B_BATCH_SIZE],
                    metadata[i : i + PHASE_B_BATCH_SIZE],
                )

            index_dir = output_path / backend / f"shard_{shard_id}"
            store.save(index_dir)

        index_size = _dir_size_bytes(index_dir) if index_dir.exists() else 0
        stats[backend] = {
            "duration_s": t.elapsed_s,
            "num_vectors": len(ids),
            "index_size_bytes": index_size,
        }
        record_metric(
            "index_build",
            RESULTS_PATH,
            duration_s=t.elapsed_s or 0.0,
            extra={
                "backend": backend,
                "shard_id": shard_id,
                "num_vectors": len(ids),
                "index_size_bytes": index_size,
            },
        )
        log.info(
            "index built",
            extra={"backend": backend, "shard_id": shard_id, **stats[backend]},
        )

    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/clean/paragraphs.parquet"), "--input", help="Cleaned paragraphs Parquet file."
    ),
    output: Path = typer.Option(
        Path("data/vectors/"), help="Output directory for vectors/indices."
    ),
    workers: int = typer.Option(2, help="Spark local[N] worker count."),
    num_shards: int = typer.Option(1, help="1 = centralized index; >1 = sharded by article_id."),
    backend: str = typer.Option("both", help="'faiss', 'chroma', or 'both'."),
    limit: int = typer.Option(0, help="Bound the number of paragraphs processed (0 = no limit)."),
    embedder: str = typer.Option("sentence-transformers", help="Embedder implementation to use."),
    model: str = typer.Option(DEFAULT_MODEL_NAME, help="sentence-transformers model name."),
    device: str = typer.Option(
        "cpu", help="Embedding device ('cpu' recommended for Spark benchmark runs)."
    ),
    driver_memory: str = typer.Option(
        "4g", help="Spark driver JVM memory (local mode shares this with execution)."
    ),
) -> None:
    """Compute embeddings for cleaned paragraphs via Spark and build FAISS/Chroma index(es)."""
    backends = ["faiss", "chroma"] if backend == "both" else [backend]
    for b in backends:
        if b not in ("faiss", "chroma"):
            raise typer.BadParameter(f"Unsupported backend: {b!r}")

    embeddings_dir = output / "embeddings"
    output.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("semantica-embed")
        .master(f"local[{workers}]")
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        log.info(
            "starting embedding stage",
            extra={"workers": workers, "num_shards": num_shards, "limit": limit, "model": model},
        )
        with Timer() as embed_timer:
            num_paragraphs = run_embedding_stage(
                spark,
                input_path,
                embeddings_dir,
                workers=workers,
                num_shards=num_shards,
                limit=limit,
                embedder_name=embedder,
                model_name=model,
                device=device,
            )
    finally:
        spark.stop()

    embed_throughput = num_paragraphs / embed_timer.elapsed_s if embed_timer.elapsed_s else 0.0
    record_metric(
        "embed",
        RESULTS_PATH,
        duration_s=embed_timer.elapsed_s or 0.0,
        throughput=embed_throughput,
        throughput_unit="paragraphs/s",
        worker_count=workers,
        extra={
            "num_shards": num_shards,
            "limit": limit,
            "device": device,
            "model": model,
            "embedder": embedder,
            "num_paragraphs": num_paragraphs,
        },
    )
    log.info(
        "embedding stage complete",
        extra={
            "duration_s": embed_timer.elapsed_s,
            "throughput": embed_throughput,
            "num_paragraphs": num_paragraphs,
        },
    )

    all_stats = {}
    with Timer() as index_timer:
        for shard_id in range(num_shards):
            all_stats[shard_id] = build_indices_for_shard(
                embeddings_dir, output, shard_id, backends=backends
            )

    num_sample_queries = write_sample_queries(embeddings_dir, output)
    log.info("wrote sample queries for benchmark_query.py", extra={"count": num_sample_queries})

    meta = {
        "num_paragraphs": num_paragraphs,
        "num_shards": num_shards,
        "backends": backends,
        "workers": workers,
        "model": model,
        "device": device,
        "embedder": embedder,
        "embed_duration_s": embed_timer.elapsed_s,
        "index_build_duration_s": index_timer.elapsed_s,
        "shard_stats": all_stats,
        "num_sample_queries": num_sample_queries,
    }
    (output / "generate_embeddings.meta.json").write_text(json.dumps(meta, indent=2, default=str))

    typer.echo(
        f"Embedded {num_paragraphs:,} paragraphs across {num_shards} shard(s), "
        f"backends={backends}. Output: {output}"
    )
    typer.echo(f"Metadata: {output / 'generate_embeddings.meta.json'}")


if __name__ == "__main__":
    app()
