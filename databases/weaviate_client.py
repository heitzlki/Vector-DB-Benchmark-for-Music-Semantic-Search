from typing import List, Dict, Any
import weaviate
from weaviate.classes import config as wvc
from weaviate.classes.config import Property, DataType
from weaviate.classes.data import DataObject
from .base import VectorDB

import warnings

# There is an open issue for this weaviate warning:
# https://github.com/weaviate/weaviate-python-client/issues/1416
warnings.filterwarnings(
    "ignore",
    message="Con004: The connection to Weaviate was not closed properly. This can lead to memory leaks.",
    category=ResourceWarning,
    module="weaviate.warnings",
)
warnings.filterwarnings(
    "ignore",
    message=".*unclosed.*",
    category=ResourceWarning,
)


class WeaviateDB(VectorDB):
    def _ensure_connected(self):
        # If client is closed, reconnect
        try:
            if not self.client.is_connected():
                self.client.connect()
        except Exception:
            # If client is closed (WeaviateClosedClientError), re-instantiate
            self.client = weaviate.connect_to_local(
                host=self.url.replace("http://", "").split(":")[0],
                port=int(self.url.split(":")[-1]),
                grpc_port=50051,
            )
            self.client.connect()
        # Always refresh collection handle
        self.col = self.client.collections.get(self.class_name)

    def close(self):
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
        self.col = None

    def __init__(self, url: str = "http://localhost:8080", class_name: str = "Track"):
        self.url = url
        self.class_name = class_name
        self.client = weaviate.connect_to_local(
            host=url.replace("http://", "").split(":")[0],
            port=int(url.split(":")[-1]),  # REST
            grpc_port=50051,  # must match your docker mapping
        )
        self.col = None

    def setup(self, dim: int):
        self._ensure_connected()
        # clean if exists
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(self.class_name)

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
            vector_config=wvc.Configure.Vectors.self_provided(
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    ef_construction=128,
                    max_connections=64,
                    ef=128,
                    vector_cache_max_objects=100_000,
                    distance_metric=wvc.VectorDistances.COSINE,
                )
            ),
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
                    props[k] = p.get("row_id", i)
                else:
                    val = p.get(k)
                    props[k] = "" if val is None else str(val)
            objs.append(DataObject(properties=props, vector=vectors[i]))

        # ---- chunk to stay under gRPC 100MB message cap ----
        # crude size estimate per object: vector bytes + ~512B overhead for props
        dim = len(vectors[0]) if vectors else 0
        bytes_per_obj = dim * 4 + 512
        BATCH = 2000  # standardized batch size
        for start in range(0, len(objs), BATCH):
            chunk = objs[start : start + BATCH]
            self.col.data.insert_many(chunk)

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:
        self._ensure_connected()
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
