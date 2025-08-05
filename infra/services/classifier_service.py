
from typing import Tuple, Optional, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from domain.interfaces.llm_service import LLMService
from infra.prompts.prompt_service import LocalPromptService
from infra.services.llm_service import OllamaGemmaLLMService


class SLMClassifierService(LLMService):
    """Classificador VADER leve com léxico PT/EN estendido."""
    EXTRA_LEXICON = {
        
        "excelente": 3.5, "maravilhoso": 3.5, "ótimo": 3.0,
        "bom": 2.0, "fantástico": 3.5, "awesome": 3.5, "great": 3.0,
        "péssimo": -3.5, "horrível": -3.5, "lixo": -3.0,
        "ruim": -2.5, "trash": -3.0, "terrible": -3.5, "awful": -3.5,
    }

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(self.EXTRA_LEXICON)

    def classify_with_score(self, text: str) -> Tuple[str, float]:
        comp = self.analyzer.polarity_scores(text)["compound"]
        label = "positivo" if comp >= 0.05 else "negativo" if comp <= -0.05 else "neutro"
        return label, comp

    # métodos exigidos pela interface
    def classify_text(self, text: str) -> str:
        return self.classify_with_score(text)[0]

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        return "Serviço de classificação."  # não usado aqui


class LLMClassifierWrapper(LLMService):
    """Envolve o Gemma via Ollama apenas para classificação (sem score)."""

    def __init__(self) -> None:
        prompt_service = LocalPromptService()
        self.llm = OllamaGemmaLLMService(prompt_service, model = 'gemma3:4b')

    def classify_text(self, text: str) -> str:
        return self.llm.classify_text(text)

    # exigido pela interface, mas não usado
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        return "Serviço de classificação."

    def classify_with_score(self, text: str) -> Tuple[str, None]:
        return self.classify_text(text), None


class ClassifierService:
    """
    Fachada que escolhe entre SLM (VADER) ou LLM (Gemma).
    """

    def __init__(self) -> None:
        self.slm = SLMClassifierService()
        self.llm = LLMClassifierWrapper()

    def classify(
        self,
        text: str,
        model: str = "slm"
    ) -> Tuple[str, Optional[float]]:
        model = model.lower()
        if model == "llm":
            return self.llm.classify_with_score(text)
        return self.slm.classify_with_score(text)
