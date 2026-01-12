import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from app.embedder import TicketEmbedder
from app.faiss_index import TicketFAISSIndex
from app.llm_client import generate_final_answer



class RAGPipeline:
    def __init__(self, data_path: str):
        self.embedder = TicketEmbedder()

        df = pd.read_csv(data_path)

        # 🔹 Retrieval text
        self.texts = (
            df["title"].fillna("") + " " +
            df["description"].fillna("") + " " +
            df["error_logs"].fillna("")
        ).tolist()

        # 🔹 Final answers
        self.resolutions = df["resolution"].fillna("").tolist()

        # ===== FAISS =====
        embeddings = self.embedder.embed(self.texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index = TicketFAISSIndex(embedding_dim=embeddings.shape[1])
        self.index.add_embeddings(embeddings)

        # ===== BM25 =====
        tokenized_texts = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)

        print("Pipeline ready | Records:", len(self.texts))

    def resolve_ticket(self, query: str, top_k: int = 5):
        # --- FAISS search ---
        query_embedding = self.embedder.embed([query])
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        indices, scores = self.index.search(query_embedding, top_k=top_k)

        if indices is None or len(indices[0]) == 0:
            return {
            "final_answer": "No relevant resolution found.",
            "matches": []
            }


        # Convert FAISS indices to int
        candidate_indices = [int(i) for i in indices[0]]

        # --- BM25 re-ranking ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Re-rank FAISS candidates using BM25
        reranked_indices = sorted(
            candidate_indices,
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        # Map indices to candidate resolutions
        candidates = [
            {
                "index": i,
                "text": self.texts[i],
                "resolution": self.resolutions[i]
            }
            for i in reranked_indices[:top_k]
        ]
        
        resolution_texts = [c["resolution"] for c in candidates] 

        final_answer = generate_final_answer(
          query=query,
          candidates=resolution_texts
        )
        print("\n🧠 LLM FINAL ANSWER:\n", final_answer)

        return {
           "final_answer": final_answer,
           "matches": candidates
        }





# python -m uvicorn app.main:app