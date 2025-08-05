
from typing import List, Dict, Tuple
import numpy as np
from domain.interfaces.documents_repository import DocumentRepository


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VectorDBRepository(DocumentRepository):
    def __init__(self) -> None:
        self.database: Dict[str, Dict] = {}

    def store_embeddings(
        self,
        document_id: str,
        embeddings: List[List[float]],
        metadata: Dict,
    ) -> None:
        self.database[document_id] = {
            "embeddings": np.asarray(embeddings, dtype="float32"),
            "metadata": metadata,
        }

    def search_embeddings(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[np.ndarray, Dict]]:
        if not self.database:
            return []

        q = np.asarray(query_vector, dtype="float32")
        scored: List[Tuple[float, np.ndarray, Dict]] = []

        for item in self.database.values():
            for emb in item["embeddings"]:
                score = _cosine(q, emb)
                scored.append((score, emb, item["metadata"]))

        # ordena por score desc e devolve top-k
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(emb, meta) for _, emb, meta in scored[:top_k]]
