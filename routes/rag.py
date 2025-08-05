from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
from infra.prompts.prompt_service import LocalPromptService
from infra.services.llm_service import OllamaGemmaLLMService
from infra.repositories.chroma_repository import ChromaRepository
from src.rag import RAG

router = APIRouter()

prompt_service = LocalPromptService()
llm_service = OllamaGemmaLLMService(prompt_service)
chroma_repo = ChromaRepository()             # carrega / persiste automático
rag_engine = RAG(chroma_repo, llm_service)

class RagResponse(BaseModel):
    question: str
    answer: str
    chunks: List[str]

@router.get("/ask", response_model=RagResponse, summary="Pergunta via RAG")
async def ask_question(question: str = Query(..., description="Sua pergunta")):
    answer, ctx = rag_engine.answer_question(question)
    return RagResponse(question=question, answer=answer, chunks=ctx)
