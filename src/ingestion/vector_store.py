import hashlib
import logging
import torch
import asyncio

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel
from src.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# manage qdrant vector store
class VectorStoreManager:
    def __init__(self):
        # init async qdrant client
        self.client = AsyncQdrantClient("localhost", port=6333)

        logger.info("vectorStoreManager is connecting to the embedding model...")
        self.model = ModelManager.get_embed_model()


    # utils

    # hash text to int id
    def _generate_id(self, text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**12)

    # normalize sparse vectors
    def _normalize_sparse(self, sparse_dict):
        if not sparse_dict:
            return sparse_dict
        max_val = max(sparse_dict.values()) or 1.0
        return {k: float(v / max_val) for k, v in sparse_dict.items()}


    # collection management
    
    async def create_collection(self, collection_name: str):
        """create collection by user session id"""
        collections = await self.client.get_collections()
        exists = any(c.name == collection_name for c in collections.collections)

        if exists:
            logger.info(f"Collection '{collection_name}' đã tồn tại.")
            return

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000
            )
        )
        logger.info(f"Created collection: {collection_name}")

    async def delete_collection(self, collection_name: str):
        """delete collection on new upload to clear old data"""
        try:
            await self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error occurred while deleting collection {collection_name}: {e}")

    # upsert
    async def upsert_documents(self, documents, collection_name: str, batch_size: int = 64):
        """get collection name to know where to insert"""
        total = len(documents)
        logger.info(f"starting upsert of {total} chunks into '{collection_name}' (batch={batch_size})")

        total_batches = (total + batch_size - 1) // batch_size

        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            sentences = [doc.page_content for doc in batch_docs]

            try:
                output = await asyncio.to_thread(
                    self.model.encode,
                    sentences,
                    return_dense=True,
                    return_sparse=True
                )

                dense_vectors = output['dense_vecs']
                sparse_vectors = output['lexical_weights']

                points = []
                for j, doc in enumerate(batch_docs):
                    sparse_dict = self._normalize_sparse(sparse_vectors[j])
                    point_id = self._generate_id(doc.page_content)

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense_vectors[j].tolist(),
                                "text-sparse": models.SparseVector(
                                    indices=[int(k) for k in sparse_dict.keys()],
                                    values=[float(v) for v in sparse_dict.values()]
                                )
                            },
                            payload={
                                "content": doc.page_content,
                                "original_text": doc.metadata.get("original_text"),
                                "page": doc.metadata.get("page"),
                                "chunk_index": doc.metadata.get("chunk_index"),
                                "chunk_length": doc.metadata.get("chunk_length"),
                                "is_list": doc.metadata.get("is_list"),
                            }
                        )
                    )

                await self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                logger.info(f"Batch {i//batch_size + 1}/{total_batches} hoàn tất.")

            except Exception as e:
                logger.error(f"error in batch {i//batch_size + 1}: {e}", exc_info=True)


    # query

    async def search(self, query: str, collection_name: str, top_k: int = 5):
        """must specify collection name for correct search"""
        output = await asyncio.to_thread(
            self.model.encode,
            [query],
            return_dense=True,
            return_sparse=True
        )

        dense_query = output["dense_vecs"][0].tolist()
        sparse_query = self._normalize_sparse(output["lexical_weights"][0])

        response = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=[int(k) for k in sparse_query.keys()],
                        values=list(sparse_query.values())
                    ),
                    using="text-sparse",
                    limit=top_k,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=top_k,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )

        return response.points