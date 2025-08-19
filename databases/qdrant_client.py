# databases/qdrant_client.py
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, SearchParams
from .base import VectorDB

HTTP_TIMEOUT = 300.0  # seconds
BATCH_SIZE = 200  # standardized batch size
PARALLEL = 2  # 0 = auto, or small integer


class Qdrant(VectorDB):
    def close(self):
        # QdrantClient does not require explicit close, but method is needed for interface
        pass

    def __init__(self, url: str = "http://localhost:6333", collection: str = "music"):
        # Honor the configured URL; avoid hardcoding localhost.
        # Using HTTP URL keeps behavior consistent across local/remote.
        self.client = QdrantClient(url=url, timeout=HTTP_TIMEOUT)
        self.collection = collection

    def setup(self, dim: int):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        )

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        # Use the high level bulk uploader. It handles batching and retries.
        self.client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            payload=payloads,
            ids=list(range(len(vectors))),
            batch_size=BATCH_SIZE,
            parallel=PARALLEL,  # 0 picks a sensible default based on CPU
            max_retries=5,
            wait=True,  # wait for the whole upload to finish
        )

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128),
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in res]

    def teardown(self):
        pass
