"""
Pinecone client for benchmarking using Pinecone Local (API v6.x)
"""

from pinecone.grpc import PineconeGRPC, GRPCClientConfig
from pinecone import ServerlessSpec
import numpy as np


class PineconeClient:
    def print_index_stats(self, namespace="default"):
        stats = self.describe_index_stats(namespace=namespace)

    def teardown(self):
        """
        Delete the Pinecone index for a clean state.
        """
        self.delete_index()

    def search(self, query, top_k=10):
        """
        Search Pinecone index and return results in the expected format.
        """
        try:
            response = self.query(vector=query, top_k=top_k)
        except Exception as e:
            raise
        out = []
        for match in response.get("matches", []):
            out.append(
                {
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "payload": match.get("metadata", {}),
                }
            )
        return out

    def setup(self, dim: int):
        """
        Ensure the index exists with the correct dimension. If the dimension differs, recreate the index.
        """
        if self.pc.has_index(self.index_name):
            desc = self.pc.describe_index(name=self.index_name)
            if hasattr(desc, "dimension") and desc.dimension != dim:
                self.pc.delete_index(name=self.index_name)
                self.dimension = dim
                self._init_index()
                self.index = self._get_index()
        else:
            self.dimension = dim
            self._init_index()
            self.index = self._get_index()

    def __init__(
        self,
        host="http://localhost:5080",
        api_key="pclocal",
        dimension=384,
        metric="cosine",
        index_name="benchmark-index",
    ):
        self.host = host
        self.api_key = api_key
        self.dimension = dimension
        self.metric = metric
        self.index_name = index_name
        self.pc = PineconeGRPC(api_key=self.api_key, host=self.host)
        self._init_index()
        self.index = self._get_index()

    def _init_index(self):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="disabled",
                tags={"environment": "development"},
            )

    def _get_index(self):
        index_host = self.pc.describe_index(name=self.index_name).host
        return self.pc.Index(
            host=index_host, grpc_config=GRPCClientConfig(secure=False)
        )

    def upsert(self, vectors, payloads, namespace="default", batch_size=200):
        # vectors: list of lists (float), payloads: list of dicts
        # Pinecone expects: [{"id": str, "values": [...], "metadata": {...}}]
        pinecone_vectors = []
        for i, (vec, meta) in enumerate(zip(vectors, payloads)):
            # Replace None values in metadata with empty string
            clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
            # Use row_id as id if available, else fallback to index
            record_id = str(meta.get("row_id", i))
            clean_meta["row_id"] = record_id  # Ensure row_id is in metadata for recall
            pinecone_vectors.append(
                {"id": record_id, "values": vec, "metadata": clean_meta}
            )
        # Upsert in batches to avoid gRPC message size limits
        for start in range(0, len(pinecone_vectors), batch_size):
            end = start + batch_size
            self.index.upsert(vectors=pinecone_vectors[start:end], namespace=namespace)

    def query(
        self,
        vector,
        top_k=10,
        namespace="default",
        filter=None,
        include_values=False,
        include_metadata=True,
    ):
        try:
            result = self.index.query(
                namespace=namespace,
                vector=vector,
                filter=filter,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
            )
            return result
        except Exception as e:
            raise

    def describe_index_stats(self, namespace="default"):
        return self.index.describe_index_stats(namespace=namespace)

    def delete_index(self):
        self.pc.delete_index(name=self.index_name)
