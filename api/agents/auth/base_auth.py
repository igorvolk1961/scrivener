"""
Базовый класс для провайдеров авторизации.
"""

from abc import ABC, abstractmethod
from typing import Any

from api.agents.agent_definition import LLMConfig
from openai import AsyncOpenAI


class BaseAuthProvider(ABC):
    """Базовый класс для провайдеров авторизации.
    
    Провайдеры авторизации позволяют кастомизировать процесс создания
    и использования клиента AsyncOpenAI для различных LLM провайдеров.
    """

    def get_client_kwargs(self, llm_config: LLMConfig) -> dict[str, Any] | None:
        """Возвращает дополнительные параметры для client_kwargs при создании AsyncOpenAI.
        
        Args:
            llm_config: Конфигурация LLM
            
        Returns:
            Словарь с дополнительными параметрами для client_kwargs или None
        """
        return None

    def wrap_client(self, client: AsyncOpenAI, llm_config: LLMConfig) -> AsyncOpenAI:
        """Возвращает обернутый клиент для динамических заголовков или модификации поведения.
        
        Если провайдеру нужны динамические заголовки (например, обновляемый токен),
        он может вернуть wrapper, который перехватывает запросы.
        
        Args:
            client: Базовый AsyncOpenAI клиент
            llm_config: Конфигурация LLM
            
        Returns:
            AsyncOpenAI клиент (может быть обернутый или исходный)
        """
        return client
