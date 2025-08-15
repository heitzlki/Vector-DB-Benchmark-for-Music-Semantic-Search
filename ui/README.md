Music Semantic Search – Triple DB Compare
========================================

This project provides a minimal FastAPI backend and a static frontend to compare semantic search results (and latency) across three vector databases side by side.

Features
- Side-by-side results for Qdrant, Milvus, and Weaviate
- Per-database query latency shown in milliseconds
- Simple, static UI served from the backend at root path

Quick Start
1) Environment
- Ensure Python 3.9+ is available.
- Install dependencies:
  pip install fastapi uvicorn python-dotenv sentence-transformers numpy pandas
  # And whichever DB clients you need in your environment:
  # pip install qdrant-client pymilvus weaviate-client

2) Configure
- Create a .env file in the repo root (same folder as backend/ and frontend/):
  # DB connection endpoints
  QDRANT_URL=http://localhost:6333
  MILVUS_HOST=localhost
  MILVUS_PORT=19530
  WEAVIATE_URL=http://localhost:8080
  
  # Optional: if you want the API to auto-ingest on startup/first search
  # This parquet must contain columns: embedding (list[float]), track, artist, genre, seeds, text
  EMBEDDINGS_PARQUET=./data/embeddings.parquet

3) Run
- Start the API (serves frontend too):
  uvicorn backend.server:app --reload --port 8000

- Open the app:
  http://localhost:8000

Notes
- If EMBEDDINGS_PARQUET is set, the server will lazily load and upsert into each selected DB the first time it’s used. Otherwise, it assumes that your databases are already populated and ready for search.
- The query embedding uses sentence-transformers (all-MiniLM-L6-v2 by default). The first request will download the model if not cached.
- The backend expects your DB client wrappers to expose: setup(dim:int), upsert(vectors: List[List[float]], payloads: List[dict]), search(vector: List[float], top_k:int) -> List[{"payload":..., "score":...}]. This mirrors your benchmark script contract.

Project Layout
- backend/server.py      # FastAPI app, /search endpoint, serves frontend
- frontend/index.html    # UI with three result columns and latency per DB
- frontend/app.js        # Fetch logic and DOM rendering
- frontend/styles.css    # Basic styling
