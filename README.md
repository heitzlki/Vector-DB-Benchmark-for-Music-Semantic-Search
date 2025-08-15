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
