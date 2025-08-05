from abc import ABC, abstractmethod
from typing import List

class LLMService(ABC):
    """
    Interface para serviços de LLM/SLM.
    """

    @abstractmethod
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Gera uma resposta para a pergunta usando os chunks de contexto fornecidos.
        """
        raise NotImplementedError("O método 'generate_answer' precisa ser implementado.")

    @abstractmethod
    def classify_text(self, text: str) -> str:
        """
        Classifica o texto em uma categoria (ex.: sentimento positivo/negativo).
        """
        raise NotImplementedError("O método 'classify_text' precisa ser implementado.")
