from abc import ABC, abstractmethod
from typing import List, Dict

class DocumentRepository(ABC):
    """
    Interface para repositórios que armazenam embeddings e metadados.
    """

    @abstractmethod
    def store_embeddings(self, document_id: str, embeddings: List[List[float]], metadata: Dict) -> None:
        """
        Salva embeddings e metadados na base vetorial.
        """
        raise NotImplementedError("O método 'store_embeddings' precisa ser implementado.")

    @abstractmethod
    def search_embeddings(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Busca embeddings semelhantes na base vetorial.
        """
        raise NotImplementedError("O método 'search_embeddings' precisa ser implementado.")
