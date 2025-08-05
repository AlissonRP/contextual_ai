from fastapi import FastAPI
from pathlib import Path
import uuid, os

from routes import documents, rag, classifier
from infra.services.embedding_service import HFEmbeddingService
from src.documents_processing import DocumentProcessing
from infra.repositories import chroma_repo
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_app() -> FastAPI:
    app = FastAPI(title="Contextual AI API", version="1.0.0")

    app.include_router(documents.router,  prefix="/documents",  tags=["Documents"])
    app.include_router(rag.router,        prefix="/rag",        tags=["RAG"])
    app.include_router(classifier.router, prefix="/classifier", tags=["Classifier"])

    @app.get("/", summary="Status")
    async def health():
        return {"msg": "running"}

    # -------- ingestão automática ----------------------------------------
    @app.on_event("startup")
    def ingest_local_pdfs() -> None:
        docs_dir = Path("data/docs")
        if not docs_dir.exists():
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedder = HFEmbeddingService()
        processor = DocumentProcessing()   # usa chroma_repo internamente

        for pdf in docs_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for p in pages:
                p.metadata["source"] = pdf.name
            splits = splitter.split_documents(pages)
            texts  = [s.page_content for s in splits]
            metas  = [s.metadata      for s in splits]
            chroma_repo.add_texts(texts, metas)

        print(f"Adicionados {len(chroma_repo._chroma.get())}documentos em Chroma.")

    return app



app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
