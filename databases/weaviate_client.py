from typing import List, Dict, Any
import weaviate
from weaviate.classes import config as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject
from .base import VectorDB


class WeaviateDB(VectorDB):
    def __init__(self, url: str = "http://localhost:8080", class_name: str = "Track"):
        self.client = weaviate.connect_to_local(
            host=url.replace("http://", "").split(":")[0],
            port=int(url.split(":")[-1]),  # REST
            grpc_port=50051,  # must match your docker mapping
        )
        self.class_name = class_name
        self.col = None

    def setup(self, dim: int):
        # clean if exists
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)

        # 4.16.7-friendly: use builder helpers but pass them through
        # vector_index_config / vectorizer_config (no VectorConfig wrapper).
        self.client.collections.create(
            name=self.class_name,
            description="Music embeddings",
            properties=[
                Property(name="row_id", data_type=DataType.INT),
                Property(name="track", data_type=DataType.TEXT),
                Property(name="artist", data_type=DataType.TEXT),
                Property(name="genre", data_type=DataType.TEXT),
                Property(name="seeds", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            # Do NOT pass distance here; let it default to COSINE
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                ef_construction=128,
                max_connections=64,
                ef=128,
                vector_cache_max_objects=100_000,
            ),
            vectorizer_config=wvc.Configure.Vectorizer.none(),
        )
        self.col = self.client.collections.get(self.class_name)

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        # Build DataObjects with only schema fields
        allowed = {"row_id", "track", "artist", "genre", "seeds", "text"}
        objs: List[DataObject] = []
        for i, p in enumerate(payloads):
            props = {}
            for k in allowed:
                if k == "row_id":
                    props[k] = i
                else:
                    val = p.get(k)
                    props[k] = "" if val is None else str(val)
            objs.append(DataObject(properties=props, vector=vectors[i]))

        # ---- chunk to stay under gRPC 100MB message cap ----
        # crude size estimate per object: vector bytes + ~512B overhead for props
        dim = len(vectors[0]) if vectors else 0
        bytes_per_obj = dim * 4 + 512
        TARGET_MB = 50  # aim for ~50MB batches to be safe
        max_per_batch = max(
            1, min(10_000, int((TARGET_MB * 1024 * 1024) / max(1, bytes_per_obj)))
        )

        for start in range(0, len(objs), max_per_batch):
            chunk = objs[start : start + max_per_batch]
            self.col.data.insert_many(chunk)

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        if self.col is None:
            # if client was closed or not connected, reconnect
            try:
                self.client.is_connected()  # or a cheap call
            except:
                self.client = weaviate.connect_to_local(
                    host=self.url_host, port=self.url_port, grpc_port=self.grpc_port
                )
            self.col = self.client.collections.get(self.class_name)

        res = self.col.query.near_vector(
            near_vector=query, limit=top_k, return_metadata=["distance"]
        )
        out = []
        for o in res.objects:
            out.append(
                {
                    "id": o.uuid,
                    "score": float(o.metadata.distance),
                    "payload": o.properties,
                }
            )
        return out

    def teardown(self):
        self.client.close()
