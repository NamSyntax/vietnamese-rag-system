# src/retrieval/search_engine.py
import logging
import torch
import unicodedata
import asyncio
from qdrant_client import models, AsyncQdrantClient
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

from src.utils.nlp_utils import segment_vietnamese
from src.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

def remove_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

class RAGRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient("localhost", port=6333)

        logger.info("Loading BGE-M3 (Embedding)...")
        self.embed_model = ModelManager.get_embed_model()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Reranker on device: {device}...")
        self.reranker = CrossEncoder(
            'BAAI/bge-reranker-v2-m3',
            device=device
        )

    def _expand_query(self, query: str):
        queries = {
            query.strip(),
            query.lower().strip(),
            remove_accents(query).strip()
        }
        return [q for q in queries if q]

    def _normalize_sparse(self, sparse_dict):
        if not sparse_dict:
            return sparse_dict
        max_val = max(sparse_dict.values()) or 1.0
        return {int(k): float(v / max_val) for k, v in sparse_dict.items()}

    
    # Core search (Hybrid + RRF + Rerank)
    
    async def search(self, query: str, collection_name: str, top_k: int = 5, score_threshold: float = 0.0):
        """tham số collection_name để truy hồi đúng kho dữ liệu của user"""
        segmented_query = segment_vietnamese(query)
        queries = self._expand_query(segmented_query)
        all_hits = {}

        # -------- HYBRID SEARCH (Qdrant v1.16+ API) --------
        for q in queries:
            emb = await asyncio.to_thread(
                self.embed_model.encode,
                [q],
                return_dense=True,
                return_sparse=True
            )

            dense_query = emb["dense_vecs"][0].tolist()
            sparse_query = self._normalize_sparse(emb["lexical_weights"][0])

            try:
                response = await self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=list(sparse_query.keys()),
                                values=list(sparse_query.values())
                            ),
                            using="text-sparse",
                            limit=50,
                        ),
                        models.Prefetch(
                            query=dense_query,
                            using="dense",
                            limit=50,
                        )
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=50
                )
                
                for hit in response.points:
                    all_hits[hit.id] = hit
                    
            except Exception as e:
                logger.error(f"Error searching in collection '{collection_name}': {e}")
                raise RuntimeError(f"Không thể truy cập dữ liệu của session này.")

        if not all_hits:
            return []

        unique_hits = list(all_hits.values())

        # -------- RERANK ------
        passages = [
            hit.payload.get("original_text") or hit.payload.get("content", "")
            for hit in unique_hits
        ]

        rerank_results = await asyncio.to_thread(
            self.reranker.rank,
            query,
            passages,
            return_documents=True,
            top_k=min(len(passages), top_k * 4)
        )

        # -------- FILTER + DIVERSITY ------
        final_docs = []
        seen = set()

        for res in rerank_results:
            if res["score"] < score_threshold:
                continue

            hit = unique_hits[res["corpus_id"]]
            text = hit.payload.get("original_text", "")

            key = text[:200]
            if key in seen:
                continue

            final_docs.append({
                "content": text,
                "score": float(res["score"]),
                "page": hit.payload.get("page"),
                "chunk_index": hit.payload.get("chunk_index"),
            })

            seen.add(key)

            if len(final_docs) >= top_k:
                break

        logger.info(f"retrieved {len(final_docs)} documents from collection {collection_name}")
        return final_docs