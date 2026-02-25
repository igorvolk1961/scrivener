"""
Agent Factory для динамического создания агентов из определений.
Адаптировано из sgr-agent-core.
"""

import importlib
import inspect
import logging
from datetime import datetime
from typing import Optional, Tuple, Type, TypeVar

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from api.agents.agent_definition import AgentDefinition, LLMConfig
from api.agents.base_agent import BaseAgent
from api.agents.auth.base_auth import BaseAuthProvider
from api.agents.registry import AgentRegistry, ToolRegistry
from loguru import logger

Agent = TypeVar("Agent", bound=BaseAgent)


class AgentFactory:
    """Factory for creating agent instances from definitions.

    Use AgentRegistry and ToolRegistry to look up agent classes by name
    and create instances with the appropriate configuration.
    """

    @classmethod
    def _load_auth_provider(cls, auth_module: str):
        """Загружает провайдер авторизации из указанного модуля.
        
        Args:
            auth_module: Путь к модулю (например, 'api.agents.auth.gigachat_auth')
                        или полный путь с классом (например, 'api.agents.auth.gigachat_auth.GigaChatAuthProvider')
            
        Returns:
            Экземпляр провайдера авторизации
            
        Raises:
            ImportError: Если модуль или класс не найден
            AttributeError: Если класс провайдера не найден в модуле
        """
        # Определяем имя класса провайдера
        # Если указан полный путь с классом (например, 'api.agents.auth.gigachat_auth.GigaChatAuthProvider'),
        # используем его; иначе ищем в модуле класс-наследник BaseAuthProvider с именем *AuthProvider
        parts = auth_module.rsplit('.', 1)
        if len(parts) == 2 and parts[1] and parts[1][0].isupper():
            module_path = parts[0]
            class_name = parts[1]
        else:
            module_path = auth_module
            class_name = None  # будем искать в модуле
        
        module = importlib.import_module(module_path)
        
        if class_name is not None:
            provider_class = getattr(module, class_name)
            if not inspect.isclass(provider_class):
                raise TypeError(
                    f"'{class_name}' в модуле '{module_path}' не является классом. "
                    f"Получен тип: {type(provider_class)}"
                )
            return provider_class()
        
        # Ищем в модуле класс, наследующий BaseAuthProvider и оканчивающийся на AuthProvider
        for attr_name in dir(module):
            if attr_name.endswith('AuthProvider') and attr_name[0].isupper():
                obj = getattr(module, attr_name)
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseAuthProvider)
                    and obj is not BaseAuthProvider
                ):
                    return obj()
        
        attrs = [x for x in dir(module) if not x.startswith('_')]
        raise AttributeError(
            f"В модуле '{module_path}' не найден класс-наследник BaseAuthProvider. "
            f"Доступные атрибуты: {attrs}"
        )

    @classmethod
    def create_client(cls, llm_config: LLMConfig) -> Tuple[AsyncOpenAI, Optional[datetime]]:
        """Create OpenAI client from configuration.
        
        Этот метод можно использовать для создания клиента OpenAI как для агентов,
        так и для прямого использования LLM без агентов. Поддерживает кастомные
        провайдеры авторизации через auth_module.

        Args:
            llm_config: LLM configuration

        Returns:
            (client, expires_at): клиент и момент истечения токена (None если ключ не устаревает)

        Examples:
            >>> from api.agents.agent_definition import LLMConfig
            >>> llm_config = LLMConfig(
            ...     api_key="...",
            ...     base_url="https://api.openai.com/v1",
            ...     model="gpt-4",
            ...     auth_module="api.agents.auth.gigachat_auth"  # опционально
            ... )
            >>> client = AgentFactory.create_client(llm_config)
            >>> response = await client.chat.completions.create(...)
        """
        client_kwargs = {"base_url": llm_config.base_url, "api_key": llm_config.api_key}
        if llm_config.proxy:
            client_kwargs["http_client"] = httpx.AsyncClient(proxy=llm_config.proxy)

        # Если указан auth_module, загружаем провайдер: api_key и доп. параметры (заголовки и т.д.)
        if llm_config.auth_module:
            try:
                provider = cls._load_auth_provider(llm_config.auth_module)
                # api_key: провайдер может вернуть текущий токен (постоянный ключ → временный); иначе используем llm_config.api_key
                if hasattr(provider, 'get_api_key_for_client'):
                    key = provider.get_api_key_for_client(llm_config)
                    if key is not None:
                        client_kwargs["api_key"] = key
                if hasattr(provider, 'get_client_kwargs'):
                    extra_kwargs = provider.get_client_kwargs(llm_config)
                    if extra_kwargs:
                        client_kwargs.update(extra_kwargs)
                # GigaChat и прокси gpt2giga часто работают по HTTPS с самоподписанным сертификатом — принудительно отключаем проверку SSL
                if "gigachat_auth" in (llm_config.auth_module or ""):
                    client_kwargs["http_client"] = httpx.AsyncClient(verify=False, timeout=60.0)
                client = AsyncOpenAI(**client_kwargs)
                if hasattr(provider, 'wrap_client'):
                    client = provider.wrap_client(client, llm_config)
                expires_at = provider.get_token_expires_at() if hasattr(provider, 'get_token_expires_at') else None
                return (client, expires_at)
            except Exception as e:
                logger.error(f"Ошибка при загрузке auth provider '{llm_config.auth_module}': {e}", exc_info=True)
                raise ValueError(f"Failed to load auth provider '{llm_config.auth_module}': {e}") from e

        # Стандартная логика без изменений (ключ не устаревает)
        return (AsyncOpenAI(**client_kwargs), None)

    @classmethod
    async def create(cls, agent_def: AgentDefinition, task_messages: list[ChatCompletionMessageParam]) -> Agent:
        """Create an agent instance from a definition.

        Args:
            agent_def: Agent definition with configuration (classes already resolved)
            task_messages: Task messages in OpenAI ChatCompletionMessageParam format

        Returns:
            Created agent instance

        Raises:
            ValueError: If agent creation fails
        """
        BaseClass: Type[Agent] | None = (
            AgentRegistry.get(agent_def.base_class) if isinstance(agent_def.base_class, str) else agent_def.base_class
        )
        if BaseClass is None:
            error_msg = (
                f"Agent base class '{agent_def.base_class}' not found in registry.\n"
                f"Available base classes: {', '.join([c.__name__ for c in AgentRegistry.list_items()])}\n"
                f"To fix this issue:\n"
                f"  - Check that '{agent_def.base_class}' is spelled correctly in your configuration\n"
                f"  - Ensure the custom agent classes are imported before creating agents "
                f"(otherwise they won't be registered)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # MCP tools пока не поддерживаются
        # mcp_tools: list = await MCP2ToolConverter.build_tools_from_mcp(agent_def.mcp)

        tools = []
        for tool in agent_def.tools:
            if isinstance(tool, str):
                tool_class = ToolRegistry.get(tool)
                if tool_class is None:
                    error_msg = (
                        f"Tool '{tool}' not found in registry.\nAvailable tools: "
                        f"{', '.join([c.__name__ for c in ToolRegistry.list_items()])}\n"
                        f"  - Ensure the custom tool classes are imported before creating agents "
                        f"(otherwise they won't be registered)"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                tool_class = tool
            tools.append(tool_class)

        try:
            agent = BaseClass(
                task_messages=task_messages,
                def_name=agent_def.name,
                toolkit=tools,
                openai_client=cls.create_client(agent_def.llm)[0],
                agent_config=agent_def,
            )
            logger.info(
                f"Created agent '{agent_def.name}' "
                f"using base class '{BaseClass.__name__}' "
                f"with {len(agent.toolkit)} tools"
            )
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent '{agent_def.name}': {e}", exc_info=True)
            raise ValueError(f"Failed to create agent: {e}") from e
