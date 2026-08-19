# embeddings/ — Part 2: Spark Processing

**Status: implemented.** Reads `data/clean/paragraphs.parquet`, computes embeddings via a Spark job, and builds FAISS/ChromaDB vector store indices, instrumented via [`benchmarks/bench_common/`](../benchmarks/bench_common/).

## Modules

| File | Purpose |
|---|---|
| `embedder.py` | `Embedder` ABC + `SentenceTransformerEmbedder` (local model, default `all-MiniLM-L6-v2`). Interface designed so an external embedding API can plug in later without touching the Spark job. |
| `vector_store/base.py` | `VectorStore` Protocol + `SearchResult` — the common interface both backends implement. |
| `vector_store/faiss_store.py` | `FaissVectorStore` — exact cosine search (`IndexFlatIP`), string ids mapped to FAISS's required int64 via a stable hash. |
| `vector_store/chroma_store.py` | `ChromaVectorStore` — wraps `chromadb.PersistentClient`; native string ids. |
| `sharding.py` | `assign_shard(article_id, num_shards)` — pure function; `num_shards=1` is just the degenerate single-shard (centralized) case of the same code path. |
| `generate_embeddings.py` | Main CLI: Spark embedding stage + single-process index build. |
| `benchmark_query.py` | CLI: centralized-vs-sharded query latency benchmark. |

## Why two phases

FAISS/Chroma index construction is inherently single-process — Spark executors can't concurrently mutate one shared index. So `generate_embeddings.py` does:

- **Phase A (Spark, parallel — the benchmarkable part)**: reads `paragraphs.parquet`, assigns `shard_id = article_id % num_shards` (plain arithmetic, independently recomputable by any future component), repartitions for real parallelism, and computes embeddings via `mapInPandas` — the `Embedder` is constructed **once per partition** and batch-encodes, never reloaded per row. Output is written as Hive-partitioned Parquet (`shard_id=N/`).
- **Phase B (single-process, driver, after the Spark stage)**: reads each shard's embeddings back and builds/saves its FAISS/Chroma index directly.

Run via `uv run python embeddings/generate_embeddings.py --workers N ...` (consistent with every script in this project). `spark-submit embeddings/generate_embeddings.py -- --workers N ...` also works, matching `CLAUDE.md`'s placeholder command, since the script self-constructs its `SparkSession`.

## Known gotchas (found and worked around this session)

- **JDK compatibility**: PySpark needs a JVM. The system's default `java` is version 26, which Spark 4.2 doesn't support (`ClassNotFoundException: jdk.internal.ref.Cleaner` at startup). Fix: `brew install openjdk@17` (keg-only — doesn't touch the system default) and scope `JAVA_HOME` to just the Spark invocation:
  ```bash
  JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    uv run python embeddings/generate_embeddings.py ...
  ```
- **FAISS + sentence-transformers crash if loaded in the same process.** `faiss` and `torch` (pulled in by `sentence-transformers`) link conflicting OpenMP runtimes — on this platform, having both active in one process causes a hard `SIGSEGV`, not a catchable exception. `KMP_DUPLICATE_LIB_OK=TRUE` and import-order changes were tried and don't reliably fix it — they just move where the crash happens. The fix is architectural, not a workaround flag: **`embedder.py` lazily imports `sentence_transformers` only inside `SentenceTransformerEmbedder.__init__`**, which is only ever called from within Spark executor subprocesses (`generate_embeddings.py`'s `_make_embed_partition_fn`), never from the driver process that also touches FAISS in Phase B. `benchmark_query.py` follows the same rule: it never imports `sentence_transformers` at all — `generate_embeddings.py` precomputes a handful of `sample_queries.parquet` vectors (reusing already-computed embeddings, free) specifically so the query-benchmark script only ever needs to load precomputed vectors + a vector store, never both libraries at once. **If you add a new script that touches both FAISS and the embedder, keep them in separate processes.**
- **`FutureWarning: PySpark does not yet fully support pandas >= 3.0.0`** — cosmetic, `mapInPandas` was verified working correctly with the installed `pandas==3.0.5`. Harmless.
- **Device selection**: `SentenceTransformerEmbedder` defaults to auto-selecting the best backend (would pick Apple Silicon MPS here), but `generate_embeddings.py`'s CLI defaults `--device cpu`. `local[N]` spawns N separate Python worker processes; letting each auto-select MPS would have them contend for one shared GPU, polluting the worker-count-scaling benchmark with GPU-contention noise instead of clean CPU-core parallelism. Use `--device cpu` for any real worker-scaling benchmark run.
- **Chroma is always approximate (HNSW)**; FAISS's `IndexFlatIP` here is exact. A FAISS-vs-Chroma latency comparison is therefore not purely a sharding-strategy comparison — it's also exact-vs-approximate search. Don't misread a latency gap between backends as purely a sharding effect.

## Design notes

- **FAISS id mapping**: `paragraph_id` (e.g. `"12345_2"`) is a string; FAISS's `IndexIDMap` needs int64. `FaissVectorStore` derives the int64 id via a stable `blake2b` hash (not a Spark-assigned sequential counter), so the mapping is reproducible across separate runs / partial re-embeds with no cross-partition coordination. A side `id_map.parquet` maps the hashed id back to `paragraph_id` + metadata (FAISS itself only ever returns ids and scores). Collision probability at ~1.5M items over 64 bits is ≈6.6×10⁻⁸ — negligible, but `add_batch` still asserts no collision rather than silently overwriting. ChromaDB needs none of this; it accepts `paragraph_id` directly.
- **FAISS index type**: `IndexFlatIP` (exact, cosine via normalized inner product) — at this corpus's scale (~1.5M × 384-dim), flat search is trivial (~600M FLOPs/query). No approximate index (IVF/HNSW) is needed for correctness or the benchmarking goals here; documented as a future optimization if corpus size grows much further, not built now.
- **Vector store metadata is kept lean** (`article_id`, `article_title`) rather than duplicating full paragraph text — a future Part 3 API cross-references `data/clean/paragraphs.parquet` by `paragraph_id` for full text/context, keeping the vector store and the cleaned-data store loosely coupled per `CLAUDE.md`'s conventions.

## Running it

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
uv run python embeddings/generate_embeddings.py \
  --input data/clean/paragraphs.parquet --output data/vectors/ \
  --workers 2 --num-shards 1 --backend both --limit 5000 \
  --model all-MiniLM-L6-v2 --device cpu

uv run python embeddings/benchmark_query.py --vectors-dir data/vectors/ --backend faiss --k 10
```

For the full corpus (no `--limit`), the command that actually produced the benchmark numbers below:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
uv run python embeddings/generate_embeddings.py \
  --input data/clean/paragraphs.parquet --output data/vectors/full_sharded_4/ \
  --workers 8 --num-shards 4 --backend both \
  --model all-MiniLM-L6-v2 --device cpu --driver-memory 6g
```

This is a multi-hour, multi-GB operation (see below) — run it in the background. The corresponding centralized index was built *without* a second embedding pass, by reading all 4 shards' already-computed embeddings back and building one merged index from their union (see git history for the exact one-off script — this isn't wired into the CLI since it's a one-time comparison step, not a repeated workflow).

Run `--help` on either script for the full documented CLI reference.

Every stage appends one JSON line per run to `benchmarks/results/part2_metrics.jsonl` via `benchmarks/bench_common/metrics.py` (stages: `embed`, `index_build`, `query_centralized`/`query_sharded`) — same shared envelope Part 1 uses in `part1_metrics.jsonl`.

**Important for comparing configurations**: point different `--num-shards`/`--backend` runs at *different* `--output` directories (e.g. `data/vectors/full_centralized/` vs `data/vectors/full_sharded_4/`, as used below). Both configurations write to `shard_0`/`embeddings/` paths that would otherwise collide and overwrite each other if aimed at the same output directory.

## Benchmark results (full 1,556,952-paragraph corpus)

All numbers below are from real runs on this machine (10-core Apple Silicon, `--device cpu`), logged in `benchmarks/results/part2_metrics.jsonl`. Full corpus ≈ 2.4GB of raw embedding data (1,556,952 × 384-dim float32).

**Worker-count scaling** (`--limit 20000`, single centralized shard, FAISS only — isolates the Spark stage from index-build cost):

| Workers | Throughput (paragraphs/s) |
|---|---|
| 1 | 74.9 |
| 2 | 124.5 |
| 4 | **129.3** ← peak |
| 8 | 119.9 ← regression |

Scaling is strong 1→2 workers, nearly flat 2→4, and **regresses** at 8. This is thread oversubscription, not a Spark or code problem: `sentence-transformers`/`torch` internally multithreads each `.encode()` call (via BLAS/MKL); with 8 concurrent Spark worker *processes* each also spinning up their own internal thread pool, they contend for the same 10 physical cores instead of cleanly parallelizing. A worthwhile follow-up: pin each worker's internal thread count to 1 (e.g. `torch.set_num_threads(1)` inside `SentenceTransformerEmbedder.__init__`) so Spark's process-level parallelism is the only parallelism axis, and re-run the sweep for a cleaner scaling curve.

**Full-corpus run** (`--workers 8`, `--num-shards 4`, both backends):
- **Embedding**: 2h 43m, **158.7 paragraphs/s** aggregate — beats the 20k-sample's best number, since fixed per-partition costs (model loading) amortize much better over 1.56M rows than 20k.
- **Index build time**, same data, per ~390k-vector shard: **FAISS ~2s, Chroma 95-171s**. Centralized (all 1.56M vectors, built by merging the 4 shards' already-computed embeddings — no re-embedding needed): **FAISS 14.7s, Chroma 19.8 minutes**. HNSW graph construction is dramatically more expensive than a flat index — barely visible at 5,000 vectors, unmissable at 1.5M.
- **Index size**: FAISS centralized 2.44GB, Chroma centralized 3.22GB.

**Query latency** (10 sample queries, k=10):

| | FAISS centralized | FAISS sharded (4) | Chroma centralized | Chroma sharded (4) |
|---|---|---|---|---|
| mean | 25.2ms | 25.2ms | 298.1ms | 221.2ms |
| p50 | 23.7ms | 23.5ms | 6.8ms | 25.5ms |
| p95 | 39.1ms | 43.6ms | 2921.7ms | 1985.6ms |

Two findings: **FAISS is essentially unaffected by sharding** at this query volume — flat search cost scales with total vector count regardless of how it's partitioned, and fan-out+merge overhead across just 4 shards is negligible. **Chroma shows huge variance** — p50 in single-digit-to-tens of ms but p95 in the *seconds* — a handful of queries hit something expensive (plausibly HNSW-internal behavior or cold-cache effects on first access). Worth treating with some caution: 10 queries is a small sample for a p95 estimate, so this is a signal worth investigating further (e.g. a larger `sample_queries.parquet`) before treating it as a settled conclusion, not proof of a specific root cause.

## Tests

```bash
uv run pytest embeddings/tests -q
```

All tests are fixture-based / synthetic-data, no network access and no real model download:

- `test_vector_store_faiss.py` — add/search/save/load round-trip, brute-force cosine-similarity cross-check, duplicate-id collision guard, `k > ntotal` handling.
- `test_vector_store_chroma.py` — same shape against a `tmp_path` `PersistentClient`.
- `test_sharding.py` — `assign_shard()` determinism, range, single-shard degenerate case.
- `test_embedder_interface.py` — the `Embedder` ABC contract via a `FakeEmbedder`, not a real model.

The real `sentence-transformers` model load (one-time ~90MB download), real `SparkSession` startup, and the actual `mapInPandas` distributed path are **not** unit tested — validated by the manual end-to-end run above, mirroring Part 1's testing philosophy.
