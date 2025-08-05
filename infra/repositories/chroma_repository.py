from typing import List, Dict
from langchain_chroma import Chroma
from infra.services.embedding_service import HFEmbeddingService

class ChromaRepository:
    """
    Armazena documentos + embeddings no ChromaDB local (./vector_store/chroma)
    """
    def __init__(self, persist_dir: str = "vector_store/chroma") -> None:
        self.embedder = HFEmbeddingService()
        self._chroma = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedder.embedder,  # LangChain Embeddings
        )

    # -------- interface usada pelo RAG -----------------
    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> None:
        self._chroma.add_texts(texts=texts, metadatas=metadatas)

    def as_retriever(self, k: int = 3):
        return self._chroma.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
