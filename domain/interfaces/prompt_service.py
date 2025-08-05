from abc import ABC, abstractmethod
from typing import Dict

class PromptService(ABC):
    """
    Interface para carregamento e renderização de prompts dinâmicos.
    """

    @abstractmethod
    def load_prompt(self, name: str) -> str:
        """
        Carrega o conteúdo bruto de um prompt pelo nome.
        """
        raise NotImplementedError("O método 'load_prompt' precisa ser implementado.")

    @abstractmethod
    def render_prompt(self, name: str, variables: Dict[str, str]) -> str:
        """
        Renderiza um prompt com variáveis substituídas.
        """
        raise NotImplementedError("O método 'render_prompt' precisa ser implementado.")
