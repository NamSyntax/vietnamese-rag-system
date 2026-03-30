import logging
import torch
import unicodedata
import asyncio
import numpy as np
from qdrant_client import models, AsyncQdrantClient
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

from src.utils.nlp_utils import segment_vietnamese
from src.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# remove text accents
def remove_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

# main rag retriever class
class RAGRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient("localhost", port=6333)
        logger.info("Loading BGE-M3 (Embedding)...")
        self.embed_model = ModelManager.get_embed_model()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading Reranker on device: {device}...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device)

    def _expand_query(self, query: str):
        queries = {query.strip(), query.lower().strip(), remove_accents(query).strip()}
        return list(q for q in queries if q) # return list to keep index order

    def _normalize_sparse(self, sparse_dict):
        if not sparse_dict:
            return sparse_dict
        max_val = max(sparse_dict.values()) or 1.0
        return {int(k): float(v / max_val) for k, v in sparse_dict.items()}

    # async hybrid search
    async def search(self, query: str, collection_name: str, top_k: int = 5, score_threshold: float = 0.0):
        segmented_query = segment_vietnamese(query)
        queries = self._expand_query(segmented_query)
        all_hits = {}

        # batch encode simultaneously
        emb = await asyncio.to_thread(
            self.embed_model.encode,
            queries,
            return_dense=True,
            return_sparse=True
        )
        
        # async search parallel qdrant requests
        search_tasks = []
        for i, q in enumerate(queries):
            dense_query = emb["dense_vecs"][i].tolist()
            sparse_query = self._normalize_sparse(emb["lexical_weights"][i])
            
            task = self.client.query_points(
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
            search_tasks.append(task)

        try:
            # receive results in parallel
            results = await asyncio.gather(*search_tasks)
            for response in results:
                for hit in response.points:
                    all_hits[hit.id] = hit
        except Exception as e:
            logger.error(f"Error searching in collection '{collection_name}': {e}")
            raise RuntimeError("Không thể truy cập dữ liệu của session này.")

        if not all_hits:
            return []

        unique_hits = list(all_hits.values())

        # rerank
        passages = [hit.payload.get("original_text") or hit.payload.get("content", "") for hit in unique_hits]
        rerank_results = await asyncio.to_thread(
            self.reranker.rank,
            query,
            passages,
            return_documents=True,
            top_k=min(len(passages), top_k * 4) # get more for filter buffer
        )

        # semantic diversity filter
        final_docs = []
        selected_embs = []
        
        # get dense vectors to calculate similarity
        top_candidates_idx = [res["corpus_id"] for res in rerank_results if res["score"] >= score_threshold]
        top_texts = [unique_hits[idx].payload.get("original_text", "") for idx in top_candidates_idx]
        
        if not top_texts:
            return []

        candidate_embs = await asyncio.to_thread(
            self.embed_model.encode, top_texts, return_dense=True
        )
        dense_embs = candidate_embs["dense_vecs"]

        for idx, text in enumerate(top_texts):
            original_hit = unique_hits[top_candidates_idx[idx]]
            current_emb = dense_embs[idx]

            # add doc if no docs selected yet
            if not selected_embs:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)
                continue

            # calc cosine sim with selected docs
            sims = np.dot(selected_embs, current_emb) / (np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(current_emb))
            
            # select if sim is low
            if np.max(sims) < 0.85:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)

            if len(final_docs) >= top_k:
                break

        logger.info(f"Retrieved {len(final_docs)} diverse documents.")
        return final_docs

    # format search result
    def _format_hit(self, hit, score):
        return {
            "content": hit.payload.get("original_text", ""),
            "score": float(score),
            "page": hit.payload.get("page"),
            "chunk_index": hit.payload.get("chunk_index"),
        }