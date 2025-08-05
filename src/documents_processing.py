from pathlib import Path
from typing import List
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from infra.repositories import chroma_repo            # singleton Chroma


class DocumentProcessing:
    """
    Extrai texto do PDF → chunking → grava chunks no ChromaDB.
    O Chroma calcula embeddings usando OllamaEmbeddings (mxbai-embed-large).
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(file_path: str) -> str:
        pages: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    # ------------------------------------------------------------------ #
    def process_pdf_document(self, file_path: str) -> int:
        """
        Processa um PDF, gera chunks e grava em ChromaDB.
        Retorna a qtde. de chunks inseridos.
        """
        raw_text = self._extract_text(file_path)
        chunks = self._chunk(raw_text)
        if not chunks:
            print(f"[WARN] Nenhum texto encontrado em {file_path}")
            return 0

        metas = [
            {"source": Path(file_path).name, "chunk_index": idx}
            for idx in range(len(chunks))
        ]
        chroma_repo.add_texts(chunks, metas)
        print(f"[Chroma] +{len(chunks)} chunks de {file_path}")
        return len(chunks)
