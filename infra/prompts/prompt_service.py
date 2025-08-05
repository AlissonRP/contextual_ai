import os
from typing import Dict
from langchain.prompts import PromptTemplate
from domain.interfaces.prompt_service import PromptService

class LocalPromptService(PromptService):
    """
    Serviço para carregar e renderizar prompts armazenados localmente.
    Usa LangChain PromptTemplate para substituição de variáveis.
    """

    def __init__(self, prompt_dir: str = os.path.join(os.path.dirname(__file__), "templates")) -> None:
        self.prompt_dir = prompt_dir

    def load_prompt(self, name: str) -> str:
        """
        Carrega o conteúdo bruto de um arquivo de prompt.
        """
        file_path = os.path.join(self.prompt_dir, f"{name}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt '{name}' não encontrado em {self.prompt_dir}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def render_prompt(self, name: str, variables: Dict[str, str]) -> str:
        """
        Renderiza o prompt substituindo variáveis com LangChain PromptTemplate.
        """
        template_str = self.load_prompt(name)
        prompt_template = PromptTemplate(input_variables=list(variables.keys()), template=template_str)
        return prompt_template.format(**variables)
