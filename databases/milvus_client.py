from typing import List, Dict, Any
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from .base import VectorDB
import math


class Milvus(VectorDB):
    def __init__(
        self, host: str = "localhost", port: str = "19530", collection: str = "music"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection
        self.col = None

    @staticmethod
    def _safe_str(x, default=""):
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return str(x)

    def setup(self, dim: int):
        # Raise gRPC message limits on the client
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            max_send_message_length=256 * 1024 * 1024,  # 256 MB
            max_receive_message_length=256 * 1024 * 1024,  # 256 MB
        )
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="track", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="artist", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="seeds", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields, description="Music embeddings")
        self.col = Collection(self.collection_name, schema, consistency_level="Strong")

        # Do not create index or load yet. Insert first for efficiency.
        # Index and load will happen after inserts in upsert().
        return

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]):
        # Prepare columns
        N = len(vectors)
        ids = list(range(N))
        tracks = [self._safe_str(p.get("track", "unknown")) for p in payloads]
        artists = [self._safe_str(p.get("artist", "unknown")) for p in payloads]
        genres = [self._safe_str(p.get("genre", "unknown")) for p in payloads]
        # Use the correct key name 'seeds' to keep parity with other backends
        seeds = [self._safe_str(p.get("seeds", "")) for p in payloads]
        texts = [self._safe_str(p.get("text", "")) for p in payloads]

        # Batch inserts to avoid gRPC 64 MB cap
        BATCH = 5000  # adjust if needed
        for i in range(0, N, BATCH):
            sl = slice(i, i + BATCH)
            self.col.insert(
                [
                    ids[sl],
                    vectors[sl],
                    tracks[sl],
                    artists[sl],
                    genres[sl],
                    seeds[sl],
                    texts[sl],
                ]
            )

        self.col.flush()

        # Build HNSW index (align with other backends)
        self.col.create_index(
            field_name="vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 128},
            },
        )
        self.col.load()

    def search(self, query: List[float], top_k: int) -> List[Dict[str, Any]]:

        if self.col is None:
            connections.connect(alias="default", host=self.host, port=self.port)
            assert utility.has_collection(
                self.collection_name
            ), "Milvus collection missing"
            self.col = Collection(self.collection_name)
            self.col.load()

        res = self.col.search(
            data=[query],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["track", "artist", "genre", "seeds", "text"],
        )
        out = []
        for hits in res:
            for h in hits:
                out.append(
                    {
                        "id": h.id,
                        "score": float(h.distance),
                        "payload": {
                            "track": h.entity.get("track"),
                            "artist": h.entity.get("artist"),
                            "genre": h.entity.get("genre"),
                            "seeds": h.entity.get("seeds"),
                            "text": h.entity.get("text"),
                        },
                    }
                )
        return out

    def teardown(self):
        pass
