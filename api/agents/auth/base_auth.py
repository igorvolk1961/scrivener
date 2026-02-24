"""
Базовый класс для провайдеров авторизации.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

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

    def get_api_key_for_client(self, llm_config: LLMConfig) -> str | None:
        """Возвращает api_key для создания клиента.
        
        Если провайдер использует постоянный ключ для получения временного токена,
        возвращает текущий токен (при отсутствии — получает новый). Иначе возвращает None,
        и будет использован llm_config.api_key напрямую (он не устаревает).
        
        Args:
            llm_config: Конфигурация LLM (в api_key передаётся ключ — постоянный или временный)
            
        Returns:
            Строка для поля api_key клиента или None (использовать llm_config.api_key)
        """
        return None

    def get_token_expires_at(self) -> Optional[datetime]:
        """Момент истечения текущего токена (для провайдеров с временным токеном).
        
        Если провайдер использует временный токен, возвращает datetime истечения.
        Иначе None (ключ не устаревает). Используется для проактивного обновления клиента в кэше.
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
