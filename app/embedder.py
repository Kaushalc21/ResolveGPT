from sentence_transformers import SentenceTransformer
from typing import List, Union
import pandas as pd
import numpy as np


class TicketEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def prepare_text(row: pd.Series) -> str:
        return (
            f"Title: {row.get('title', '')}\n"
            f"Description: {row.get('description', '')}\n"
            f"Tech Stack: {row.get('tech_stack', '')}\n"
            f"Error Logs: {row.get('error_logs', '')}\n"
            f"Resolution: {row.get('resolution', '')}"
        )

    def embed(self, data: Union[pd.DataFrame, List[str]]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            texts = data.apply(self.prepare_text, axis=1).tolist()
        elif isinstance(data, list):
            texts = data
        else:
            raise TypeError("Input must be DataFrame or List[str]")

        return self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
