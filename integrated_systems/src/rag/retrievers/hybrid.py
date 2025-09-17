"""
Hybrid Retrieval combining dense and sparse methods
"""

from typing import List, Dict, Any
import numpy as np
from .dense import DenseRetriever
from .sparse import SparseRetriever

class HybridRetriever:
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(self, embedding_manager, vector_store, config):
        self.dense_retriever = DenseRetriever(embedding_manager, vector_store, config)
        self.sparse_retriever = SparseRetriever(config)
        self.alpha = 0.7  # Weight for dense vs sparse
        
    async def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid approach."""
        # Get dense results
        dense_results = await self.dense_retriever.retrieve(query, top_k * 2)
        
        # Get sparse results  
        sparse_results = await self.sparse_retriever.retrieve(query, top_k * 2)
        
        # Combine and rerank
        combined_results = self._combine_results(dense_results, sparse_results)
        
        return combined_results[:top_k]
    
    def _combine_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine dense and sparse results using reciprocal rank fusion."""
        # Create score maps
        dense_scores = {doc["id"]: doc["score"] for doc in dense_results}
        sparse_scores = {doc["id"]: doc["score"] for doc in sparse_results}
        
        # Get all unique document IDs
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_id in all_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = sparse_scores.get(doc_id, 0.0)
            
            # Reciprocal Rank Fusion
            hybrid_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score
            
            # Find the document
            doc = next((d for d in dense_results + sparse_results if d["id"] == doc_id), None)
            if doc:
                doc["score"] = hybrid_score
                hybrid_results.append(doc)
        
        # Sort by hybrid score
        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)
