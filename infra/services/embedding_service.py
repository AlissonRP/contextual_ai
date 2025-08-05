"""
Embeddings via LangChain-Ollama (modelo mxbai-embed-large)
mantém a classe HFEmbeddingService apenas como fachada.
"""
from typing import List
from langchain_community.embeddings import OllamaEmbeddings
from domain.interfaces.embedding_service import EmbeddingService


class HFEmbeddingService(EmbeddingService):  # mantém o nome original
    def __init__(self, model: str = "mxbai-embed-large") -> None:
        self.embedder = OllamaEmbeddings(model=model)

    def generate_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(text_chunks)
