from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from domain.interfaces.llm_service import LLMService
from infra.repositories.chroma_repository import ChromaRepository

SYSTEM_PROMPT = "Você é um assistente que responde perguntas com base no contexto."

USER_PROMPT = (
    "Responda em português, de forma objetiva, SEM citar imagens.\n\n"
    "Contexto:\n{context}\n\nPergunta:\n{question}"
)

_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT),
                                            ("user", USER_PROMPT)])


class RAG:
    """
    Naïve-RAG usando LangChain + Chroma.
    """

    def __init__(
        self,
        chroma_repo: ChromaRepository,
        llm_service: LLMService,
        k: int = 3,
    ) -> None:
        self.retriever = chroma_repo.as_retriever(k=k)
        self.llm = llm_service.llm  # obtém o objeto LangChain LLM interno

        # monta a chain LangChain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | _prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(
            f"Fonte: {d.metadata.get('source','?')}\n{d.page_content}"
            for d in docs
        )

    # ---------- API externa usada pela rota -----------------
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        answer = self.chain.invoke(question)
        docs = self.retriever.get_relevant_documents(question)
        chunks = [d.page_content for d in docs]
        return answer, chunks
