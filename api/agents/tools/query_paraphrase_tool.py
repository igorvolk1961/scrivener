"""
Query Paraphrase Tool для перефразирования запросов пользователя.
Использует отдельный системный промпт для анализа корпоративных запросов.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from openai import AsyncOpenAI

from api.agents.base_tool import BaseTool
from utils.date_formatter import format_current_datetime

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QueryParaphraseTool(BaseTool):
    """Refine and analyze user queries to better understand intent and create optimized search queries.
    
    This tool analyzes corporate queries to understand the hidden meaning and translate
    them into the language of internal documentation. It determines query type, expected
    document types, creates refined search queries, and evaluates if context expansion is needed.
    """

    original_query: str = Field(description="Original user query to refine")

    async def __call__(self, context: AgentContext, config: AgentConfig, **kwargs) -> str:
        """
        Выполняет перефразирование запроса пользователя.
        
        Args:
            context: Контекст агента
            config: Конфигурация агента
            
        Returns:
            JSON строка с результатами перефразирования:
            - original_intent: str
            - query_type: Literal["fact", "procedure", "comparison", "calculation", "synthesis"]
            - expected_document_types: list[str]
            - refined_queries: list[str]
            - requires_context_expansion: bool
        """
        logger.info(f"QueryParaphraseTool called for query: '{self.original_query[:50]}...'")
        
        try:
            # Загружаем системный промпт из файла
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "query_paraphrase.txt"
            
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
            system_prompt = prompt_file.read_text(encoding="utf-8")
            
            # Заменяем {userPost} в промпте через config.prompts.user_info
            user_post = ""
            if config.prompts.user_info and isinstance(config.prompts.user_info, dict):
                user_post = config.prompts.user_info.get("userPost") or ""
            
            # Замена только {userPost}, без .format(), чтобы не трогать фигурные скобки в JSON
            if user_post:
                system_prompt = system_prompt.replace("{userPost}", f"Моя должность - {user_post}.")
            else:
                # Если userPost пустой, просто удаляем placeholder
                system_prompt = system_prompt.replace("{userPost}", "")

            # Замена {currentDateTime} на текущую дату и время
            if "{currentDateTime}" in system_prompt:
                formatted_date = format_current_datetime()
                system_prompt = system_prompt.replace("{currentDateTime}", formatted_date)

            queryRefinementCount = config.prompts.queryRefinementCount

            # Заменяем {queryRefinementCount} на число уточненных запросов
            if "{queryRefinementCount}" in system_prompt:
                system_prompt = system_prompt.replace("{queryRefinementCount}", str(queryRefinementCount))

            # Создаем клиент LLM через AgentFactory для поддержки кастомных провайдеров авторизации
            from api.agents.agent_factory import AgentFactory
            openai_client = AgentFactory.create_client(config.llm)
            
            # Вызываем LLM для перефразирования запроса
            response = await openai_client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.original_query}
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                response_format={"type": "json_object"}  # Требуем JSON ответ
            )
            
            # Извлекаем ответ
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response")
            
            # Парсим JSON ответ
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")
            
            # Валидируем структуру ответа
            required_fields = ["original_intent", "query_type", "expected_document_types", "refined_queries", "requires_context_expansion"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field in response: {field}")
            
            # Валидируем query_type
            valid_query_types = ["fact", "procedure", "comparison", "calculation", "synthesis"]
            if result["query_type"] not in valid_query_types:
                raise ValueError(f"Invalid query_type: {result['query_type']}. Must be one of {valid_query_types}")
            
            # Валидируем refined_queries
            refined_queries = result.get("refined_queries", [])
            if not isinstance(refined_queries, list):
                raise ValueError("refined_queries must be a list")
            if len(refined_queries) != queryRefinementCount:
                raise ValueError(f"refined_queries must contain exactly {queryRefinementCount} queries, got {len(refined_queries)}")

            # # Добавляем в начало списка оригинальный запрос
            # refined_queries.insert(0, self.original_query)

            logger.info(f"Query refinement completed. Type: {result['query_type']}, Refined queries: {len(result['refined_queries'])}")
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.exception(f"Error in QueryParaphraseTool: {e}")
            return json.dumps({
                "error": "Ошибка при перефразировании запроса",
                "detail": str(e),
                "original_query": self.original_query
            }, ensure_ascii=False, indent=2)

