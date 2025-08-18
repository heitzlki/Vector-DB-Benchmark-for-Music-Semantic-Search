import os
from topk_sdk import Client
from topk_sdk.schema import text, f32_vector, vector_index, keyword_index, int
from topk_sdk.query import select, field, fn


class TopKClient:
    def __init__(self, region=None, api_key=None):
        self.region = region or os.getenv("TOPK_REGION", "aws-us-east-1-elastica")
        self.api_key = api_key or os.getenv("TOPK_API_KEY", None)
        self.collection = os.getenv("TOPK_COLLECTION", "benchmark")
        self.client = Client(api_key=self.api_key, region=self.region)

    def setup(self):
        # Create collection with schema matching Milvus fields, default dim=384
        dim = 384
        schema = {
            "id": int().required(),
            "vector": f32_vector(dimension=dim)
            .required()
            .index(vector_index(metric="cosine")),
            "track": text().required().index(keyword_index()),
            "artist": text().required().index(keyword_index()),
            "genre": text().index(keyword_index()),
            "seeds": text().index(keyword_index()),
            "text": text(),
        }
        try:
            self.client.collections().create(self.collection, schema=schema)
        except Exception as e:
            if "already exists" not in str(e):
                raise

    def upsert(self, vectors, payloads):
        # Upsert documents with all fields for parity, using correct TopK SDK upsert
        docs = []
        for i, (vec, meta) in enumerate(zip(vectors, payloads)):
            doc = {
                "_id": str(i),
                "id": i,
                "vector": vec,
                "track": meta.get("track", "unknown"),
                "artist": meta.get("artist", "unknown"),
                "genre": meta.get("genre", ""),
                "seeds": meta.get("seeds", ""),
                "text": meta.get("text", ""),
            }
            docs.append(doc)
    BATCH_SIZE = 200
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            self.client.collection(self.collection).upsert(batch)

    def search(self, vector, top_k=10):
        # Vector search using TopK SDK query API
        col = self.client.collection(self.collection)
        docs = col.query(
            select(
                "id",
                "track",
                "artist",
                "genre",
                "seeds",
                "text",
                vector_similarity=fn.vector_distance("vector", vector),
            ).topk(field("vector_similarity"), top_k, asc=False)
        )
        # Return in the same format as other clients
        out = []
        for d in docs:
            out.append(
                {
                    "id": d.get("id"),
                    "score": d.get("vector_similarity", 0.0),
                    "payload": {
                        "track": d.get("track"),
                        "artist": d.get("artist"),
                        "genre": d.get("genre"),
                        "seeds": d.get("seeds"),
                        "text": d.get("text"),
                    },
                }
            )
        return out

    def teardown(self):
        # Optionally delete collection (not required for benchmarking)
        pass
