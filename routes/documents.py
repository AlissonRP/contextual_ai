import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from src.documents_processing import DocumentProcessing

router = APIRouter()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class UploadResponse(BaseModel):
    status: str
    processed_documents: List[str]
    total_chunks: int
    chunk_size: int
    chunk_overlap: int

@router.post(
    "/upload",
    summary="Upload e processamento de PDFs",
    response_model=UploadResponse,
)
async def upload_and_process_documents(
    uploaded_files: List[UploadFile] = File(...),
    chunk_size: int = Form(800, description="Tamanho do chunk em tokens/caracteres"),
    chunk_overlap: int = Form(200, description="Sobreposição entre chunks"),
) -> UploadResponse:
    processor = DocumentProcessing(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    processed_docs: List[str] = []
    total_chunks = 0

    for file in uploaded_files:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        total_chunks += processor.process_pdf_document(str(file_path))
        processed_docs.append(file.filename)

    return UploadResponse(
        status="success",
        processed_documents=processed_docs,
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
