# Semantica

Semantic search over Wikipedia. Chunk Wikipedia articles into paragraphs, generate embeddings for each one using Spark, and store them in a vector store so you can search by meaning instead of keyword. On top of that, a REST API for querying and a small web frontend.

## Layout

ingestion/   Wikipedia dump download + parsing
spark/       Spark jobs, embedding generation
api/         REST API in front of the vector store
docs/        proposal, reports, notes


**Part 1 (data collection):** grab a chunk of the Wikipedia dump (500MB+), clean it up, dump it in S3.

**Part 2 (Spark):** compute embeddings per paragraph across Spark workers, write them to FAISS/ChromaDB. Most of the evaluation here is about scaling — how runtime changes with worker count and data size.

**Part 3 (API + web):** `/search`, `/similar/{doc_id}`, `/embed` endpoints in front of the vector store, plus a basic frontend to actually use it.

## Data

Data lives in S3, not in this repo (see `.gitignore`). Spark runs on class-145.cs.ucr.edu.