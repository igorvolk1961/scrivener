"""
Провайдер авторизации для GigaChat.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from api.agents.auth.base_auth import BaseAuthProvider
from api.agents.agent_definition import LLMConfig


class GigaChatAuthProvider(BaseAuthProvider):
    """Провайдер авторизации для GigaChat.
    
    Управляет временным токеном (действует 30 минут) и добавляет заголовки
    Authorization и RqUID к каждому запросу.
    """

    def __init__(self):
        self.credentials: Optional[str] = None
        self.access_token: Optional[str] = None
        self.token_obtained_at: Optional[datetime] = None
        self.scope: str = "GIGACHAT_API_PERS"
        self.timeout: int = 60

    def get_client_kwargs(self, llm_config: LLMConfig) -> dict[str, Any] | None:
        """Для GigaChat не нужны дополнительные параметры, токен добавляется динамически."""
        return None

    def wrap_client(self, client: AsyncOpenAI, llm_config: LLMConfig) -> AsyncOpenAI:
        """Возвращает wrapper для управления токеном и RqUID.
        
        Args:
            client: Базовый AsyncOpenAI клиент
            llm_config: Конфигурация LLM
            
        Returns:
            Обернутый клиент с добавлением токена и RqUID к каждому запросу
        """
        # Используем стандартный api_key как credentials для получения токена
        if not llm_config.api_key:
            raise ValueError(
                "api_key не указан в LLMConfig (нужен base64(client_id:client_secret) для GigaChat)"
            )

        self.credentials = llm_config.api_key  # api_key используется как credentials для OAuth2

        return GigaChatClientWrapper(client, self)

    def _get_access_token(self) -> str:
        """Получение токена с кэшированием и автоматическим обновлением.
        
        Returns:
            Актуальный токен доступа
            
        Raises:
            RuntimeError: Если не удалось получить токен
        """
        # Проверяем срок действия токена (30 минут = 1800 секунд)
        if self.access_token and self.token_obtained_at:
            token_age = (datetime.now() - self.token_obtained_at).total_seconds()
            if token_age < 1800:
                return self.access_token

        # Получаем новый токен через OAuth2
        token = self._get_token_from_key(self.credentials)
        self.access_token = token
        self.token_obtained_at = datetime.now()
        return token

    def _get_token_from_key(self, auth_key: str) -> str:
        """Получение токена через OAuth2 (логика из giga_embeddings.py).
        
        Args:
            auth_key: Base64-encoded Authorization Key (client_id:client_secret)
            
        Returns:
            Токен доступа
            
        Raises:
            RuntimeError: Если не удалось получить токен
        """
        token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Authorization": f"Basic {auth_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        payload = {"scope": self.scope}

        try:
            with httpx.Client(timeout=self.timeout, verify=False) as client:
                response = client.post(token_url, headers=headers, data=payload)
                response.raise_for_status()
                token_data = response.json()
                if "access_token" in token_data:
                    return token_data["access_token"]
                else:
                    raise RuntimeError(
                        f"Токен не найден в ответе OAuth2 сервера GigaChat. Доступные поля: {token_data.keys()}"
                    )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP ошибка при получении токена доступа GigaChat: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ошибка при получении токена из Authorization Key: {e}") from e

    def _generate_rquid(self) -> str:
        """Генерация RqUID для запросов."""
        return str(uuid.uuid4())


class GigaChatClientWrapper:
    """Wrapper для AsyncOpenAI, добавляющий токен и RqUID к каждому запросу."""

    def __init__(self, client: AsyncOpenAI, provider: GigaChatAuthProvider):
        self._client = client
        self._provider = provider

    def __getattr__(self, name):
        """Делегируем все атрибуты базовому клиенту."""
        return getattr(self._client, name)

    @property
    def chat(self):
        """Возвращает обернутый объект chat.completions."""
        return GigaChatChatWrapper(self._client.chat, self._provider)


class GigaChatChatWrapper:
    """Wrapper для chat.completions с добавлением заголовков."""

    def __init__(self, chat, provider: GigaChatAuthProvider):
        self._chat = chat
        self._provider = provider

    @property
    def completions(self):
        """Возвращает обернутый объект completions."""
        return GigaChatCompletionsWrapper(self._chat.completions, self._provider)


class GigaChatCompletionsWrapper:
    """Wrapper для completions с перехватом create() и stream()."""

    def __init__(self, completions, provider: GigaChatAuthProvider):
        self._completions = completions
        self._provider = provider

    async def create(self, **kwargs):
        """Добавляет токен и RqUID перед запросом."""
        extra_headers = kwargs.get("extra_headers", {})
        extra_headers["Authorization"] = f"Bearer {self._provider._get_access_token()}"
        extra_headers["RqUID"] = self._provider._generate_rquid()
        kwargs["extra_headers"] = extra_headers
        return await self._completions.create(**kwargs)

    def stream(self, **kwargs):
        """Добавляет токен и RqUID перед streaming запросом."""
        extra_headers = kwargs.get("extra_headers", {})
        extra_headers["Authorization"] = f"Bearer {self._provider._get_access_token()}"
        extra_headers["RqUID"] = self._provider._generate_rquid()
        kwargs["extra_headers"] = extra_headers
        return self._completions.stream(**kwargs)
