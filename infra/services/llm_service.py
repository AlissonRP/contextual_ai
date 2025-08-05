from typing import List
from langchain_ollama.llms import OllamaLLM           # pacote langchain-ollama ≥0.1
from domain.interfaces.llm_service import LLMService
from infra.prompts.prompt_service import LocalPromptService


class OllamaGemmaLLMService(LLMService):
    """
    Wrapper do modelo Gemma (ou qualquer modelo rodando no Ollama)
    que fornece:
      • self.llm  → objeto LangChain LLM usado no RAG
      • generate_answer()  → compatibilidade com caminhos antigos
      • classify_text()    → placeholder, se necessário futuramente
    """

    def __init__(
        self,
        prompt_service: LocalPromptService,
        model: str = "gemma3:4b",        # altere para o rótulo exato no seu Ollama
    ) -> None:
        self.prompt_service = prompt_service
        # objeto LLM da LangChain-Ollama (Chat completions)
        self.llm = OllamaLLM(model=model)

  
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        prompt = (
            "Você é um assistente especializado.\n\n"
            "Contexto:\n"
            + "\n\n".join(context_chunks[:3])
            + f"\n\nPergunta: {question}\nResposta (em português):"
        )
        return self.llm.invoke(prompt)

    
    def classify_text(self, text: str) -> str:
        prompt = (
            "Classifique o sentimento do texto abaixo como POSITIVO, "
            "NEUTRO ou NEGATIVO.  Responda apenas com a palavra.\n\n"
            f"Texto: {text}\nSentimento:"
        )
        raw = self.llm.invoke(prompt)
        label = raw.strip().lower()
        # normaliza
        if "pos" in label:
            return "positivo"
        if "neg" in label:
            return "negativo"
        return "neutro"