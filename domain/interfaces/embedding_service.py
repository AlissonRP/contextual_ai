from abc import ABC, abstractmethod
from typing import List

class EmbeddingService(ABC):
    """
    Interface para serviços de geração de embeddings.
    """

    @abstractmethod
    def generate_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de textos.
        """
        raise NotImplementedError("O método 'generate_embeddings' precisa ser implementado.")
