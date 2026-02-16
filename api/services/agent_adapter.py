"""
Адаптер для преобразования AssistantRequest в AgentDefinition.
"""

import uuid
from typing import Dict, Any, Optional

from api.models.llm_models import AssistantRequest
from api.agents.agent_definition import (
    AgentDefinition,
    LLMConfig,
    SearchConfig,
    ExecutionConfig,
    PromptsConfig,
)
from api.agents.sgr_tool_calling_agent import SGRToolCallingAgent
from api.agents.tools import (
    WebSearchTool,
    RetrievalTool,
    FinalAnswerTool,
    FinalRAGAnswerTool,
    ReasoningTool,
    QueryParaphraseTool,
    ChunkHorizontalExtensionTool,
    ChunkVerticalExtensionTool
)


def create_agent_definition_from_request(
    request: AssistantRequest,
    context: Optional[Dict[str, Any]] = None,
) -> AgentDefinition:
    """
    Создает AgentDefinition из AssistantRequest.

    Args:
        request: Запрос к ассистенту
        context: Контекст запроса (userInfo, chat_messages и т.д.)

    Returns:
        AgentDefinition для создания агента
    """
    context = context or {}
    
    # Определяем набор tools на основе параметров запроса
    tools: list[type | str] = []
    
    if request.internet:
        tools.append(WebSearchTool)
    
    if request.knowledge_base:
        tools.append(RetrievalTool)
        # Добавляем инструменты для работы с базой знаний
        tools.append(QueryParaphraseTool)  # Для перефразирования запросов
        # Инструменты расширения контекста (вызываются после RetrievalTool)
        tools.append(ChunkHorizontalExtensionTool)
        tools.append(ChunkVerticalExtensionTool)
        tools.extend([FinalRAGAnswerTool, ReasoningTool])
    else:
        tools.extend([FinalAnswerTool, ReasoningTool])
    
#    system_prompt_str = request.system_prompt if request.system_prompt else None TODO: здесь нужно проверять не общий system_prompt из запроса а специальный для агента
    system_prompt_str = None

    # Извлекаем информацию о пользователе из контекста для замены {userPost}
    user_info = context.get("userInfo")
    
    # Создаем конфигурацию LLM
    llm_config = LLMConfig(
        api_key=request.llm_api_key,
        base_url=request.llm_url,
        model=request.llm_model_name,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    # Создаем конфигурацию поиска (если нужен интернет)
    search_config = None
    if request.internet:
        if not request.search_api_key:
            raise ValueError("Для использования интернет-поиска необходимо указать search_api_key")
        search_config = SearchConfig(
            tavily_api_key=request.search_api_key,
            tavily_api_base_url=request.search_url or "https://api.tavily.com",
            max_searches=4,
            max_results=10,
            content_limit=3500,
        )
    
    # Создаем конфигурацию выполнения
    execution_config = ExecutionConfig(
        max_iterations=10,
        max_clarifications=0,  # Пока отключаем уточнения
        max_retries=3,
        logs_dir="logs",
        reports_dir="reports",
    )
    
    # Создаем конфигурацию промптов
    prompts_config = PromptsConfig(
        system_prompt_str=system_prompt_str,
        initial_user_request_str=None,  # Будет использован дефолтный
        clarification_response_str=None,  # Будет использован дефолтный
        user_info=user_info,  # Передаем информацию о пользователе для замены {userPost}
    )
    
    # Создаем AgentDefinition
    agent_def = AgentDefinition(
        name=f"agent_{uuid.uuid4()}",
        base_class=SGRToolCallingAgent,
        tools=tools,
        llm=llm_config,
        search=search_config,
        execution=execution_config,
        prompts=prompts_config,
    )
    
    return agent_def
