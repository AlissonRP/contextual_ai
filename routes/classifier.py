from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from infra.services.classifier_service import ClassifierService

router = APIRouter()
facade = ClassifierService()

class TextInput(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    text: str
    label: str
    model_used: str
    score: Optional[float] = None

@router.post(
    "/classify",
    summary="Classificação de Sentimento (SLM ou LLM)",
    response_model=ClassificationResponse
)
async def classify_text(
    input_data: TextInput,
    model: str = Query("slm", description="slm ou llm")
) -> ClassificationResponse:
    label, score = facade.classify(input_data.text, model=model)
    return ClassificationResponse(
        text=input_data.text,
        label=label,
        model_used=model.lower(),
        score=score
    )
