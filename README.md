# Vector DB Benchmark for Music Semantic Search

This repo helps you benchmark popular vector databases on the same dataset and queries. The goal is apples to apples comparison for ingest speed and query latency. You can also compute simple relevance scores using tags and genres as weak labels.

## What you will benchmark

- Insert time for the entire dataset
- Average query latency for top-k search
- Optional relevance score using a simple heuristic

## Databases supported

- Qdrant (local or cloud)
- Milvus (local)
- Weaviate (local or cloud)

## Dataset

Use the Muse Musical Sentiment dataset from Kaggle. Put the CSV in `data/` and name it `muse.csv`.
Kaggle: https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset

You can start with the included `data/sample_data.csv` for a dry run. Replace it with the full dataset before filming.

## Quick start

1. Create and activate a virtualenv
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill API keys or URLs if needed
4. Start local databases with Docker Compose where relevant
5. Run embedding script
6. Run the benchmark

### Docker Compose (local)

```
docker compose -f scripts/docker-compose.yml up -d
```

This starts Qdrant, Milvus standalone, and Weaviate with default settings.

### Generate embeddings

By default we use `sentence-transformers` `all-MiniLM-L6-v2` for reproducible local tests. You can switch to OpenAI in `embeddings/embed.py`.

```
python embeddings/embed.py --csv data/muse.csv --out data/embeddings.parquet
```

### Run benchmark

Pick which databases to test using flags.

```
python benchmark.py   --csv data/muse.csv   --embeddings data/embeddings.parquet   --dbs qdrant milvus weaviate   --topk 10   --repetitions 5
```

Results will be saved in `results/metrics.json` and `results/summary.png`.

#### Optional: Teardown After Benchmark

By default, the benchmark will **not** delete the database or index after running, so you can use the data in the backend or UI. If you want to delete (teardown) the DB/index after benchmarking, use the `--teardown_after_benchmark` flag:

```
python benchmark.py --csv data/muse.csv --embeddings data/embeddings.parquet --dbs pinecone --teardown_after_benchmark
```

This is useful if you want to ensure a clean state after benchmarking and do not need to keep the data for further use.

## Heuristic relevance

We do not have human labeled query relevance. For a light touch metric we map natural language queries to expected tags or genres, then measure hit rate in the payload of the top-k results. This is weak supervision and should be presented as such in your video.

## Notes for filming

- Show the same queries across all databases
- Reset or recreate collections between runs
- Keep hardware stable
- Show both ingest time and query latency charts
- Call out that hosted providers add network latency

## Troubleshooting

- If Docker ports conflict, change them in `scripts/docker-compose.yml`
- If dimension mismatch errors appear, confirm the embedding model and index vector size
- If OpenAI is used, set your `OPENAI_API_KEY`

---

# UI: Music Semantic Search – Triple DB Compare

This project also provides a minimal FastAPI backend and a static frontend to compare semantic search results (and latency) across three vector databases side by side.

## UI Features

- Side-by-side results for Qdrant, Milvus, and Weaviate
- Per-database query latency shown in milliseconds
- Simple, static UI served from the backend at root path

## UI Quick Start

### 1. Environment

- Ensure Python 3.9+ is available.
- Install dependencies:
  ```sh
  pip install fastapi uvicorn python-dotenv sentence-transformers numpy pandas
  # And whichever DB clients you need in your environment:
  # pip install qdrant-client pymilvus weaviate-client
  ```

### 2. Configure

- Create a `.env` file in the repo root (same folder as backend/ and frontend/):
  ```env
  # DB connection endpoints
  QDRANT_URL=http://localhost:6333
  MILVUS_HOST=localhost
  MILVUS_PORT=19530
  WEAVIATE_URL=http://localhost:8080
  # Optional: if you want the API to auto-ingest on startup/first search
  # This parquet must contain columns: embedding (list[float]), track, artist, genre, seeds, text
  EMBEDDINGS_PARQUET=./data/embeddings.parquet
  ```

### 3. Run

- Start the API (serves frontend too):
  ```sh
  uvicorn backend.server:app --reload --port 8000
  ```
- Open the app: [http://localhost:8000](http://localhost:8000)

#### Notes

- If `EMBEDDINGS_PARQUET` is set, the server will lazily load and upsert into each selected DB the first time it’s used. Otherwise, it assumes that your databases are already populated and ready for search.
- The query embedding uses sentence-transformers (`all-MiniLM-L6-v2` by default). The first request will download the model if not cached.
- The backend expects your DB client wrappers to expose: `setup(dim:int)`, `upsert(vectors: List[List[float]], payloads: List[dict])`, `search(vector: List[float], top_k:int) -> List[{"payload":..., "score":...}]`. This mirrors your benchmark script contract.

## UI Project Layout

- `backend/server.py` # FastAPI app, /search endpoint, serves frontend
- `frontend/index.html` # UI with three result columns and latency per DB
- `frontend/app.js` # Fetch logic and DOM rendering
- `frontend/styles.css` # Basic styling
