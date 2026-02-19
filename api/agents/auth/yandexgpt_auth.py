"""
Провайдер авторизации для YandexGPT.
"""

from typing import Any

from api.agents.auth.base_auth import BaseAuthProvider
from api.agents.agent_definition import LLMConfig


def extract_cloud_folder(model: str) -> str | None:
    """
    Извлекает cloud folder из строки model формата "gpt://<cloud_folder>/...".
    
    Args:
        model: Строка модели в формате gpt://<cloud_folder>/...
        
    Returns:
        Идентификатор папки или None, если формат не соответствует
    """
    prefix = "gpt://"
    if model.startswith(prefix):
        after_prefix = model[len(prefix):]
        return after_prefix.split("/", 1)[0]
    return None


class YandexGPTAuthProvider(BaseAuthProvider):
    """Провайдер авторизации для YandexGPT.
    
    Извлекает folder_id из поля model (формат gpt://<folder_id>/...)
    и добавляет его как параметр project в client_kwargs.
    """

    def get_client_kwargs(self, llm_config: LLMConfig) -> dict[str, Any] | None:
        """Возвращает дополнительные параметры для client_kwargs в AsyncOpenAI.
        
        Извлекает folder_id из llm_config.model и возвращает его как параметр project.
        
        Args:
            llm_config: Конфигурация LLM
            
        Returns:
            Словарь с параметром project или None, если folder_id не найден
        """
        folder_id = extract_cloud_folder(llm_config.model)
        if folder_id:
            return {"project": folder_id}
        return None
