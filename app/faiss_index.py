# app/faiss_index.py

import faiss
import numpy as np
from typing import Tuple


class TicketFAISSIndex:
    """
    FAISS index for semantic search over support tickets.
    Uses Inner Product similarity (cosine similarity with normalized vectors).
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.

        Args:
            embedding_dim (int): Dimension of embedding vectors (e.g. 384)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add ticket embeddings to FAISS index.

        Args:
            embeddings (np.ndarray): Shape (num_tickets, embedding_dim)
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D numpy array")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k most similar tickets.

        Args:
            query_embedding (np.ndarray): Shape (1, embedding_dim)
            top_k (int): Number of results to return

        Returns:
            indices (np.ndarray): Indices of matched tickets
            scores (np.ndarray): Similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected query embedding dimension {self.embedding_dim}, "
                f"got {query_embedding.shape[1]}"
            )
            
            
        # ✅ REQUIRED FIX (normalize query)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        return indices, scores

