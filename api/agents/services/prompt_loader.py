"""
Загрузчик промптов для агентов.
Адаптировано из sgr-agent-core.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.agents.agent_definition import PromptsConfig
    from api.agents.base_tool import BaseTool

from utils.date_formatter import format_current_datetime

# Путь к папке с промптами
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt_file(filename: str) -> str:
    """
    Загружает промпт из файла.
    
    Args:
        filename: Имя файла промпта (например, "web_search.txt")
    
    Returns:
        Содержимое файла промпта
    """
    prompt_path = _PROMPTS_DIR / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"Файл промпта не найден: {prompt_path}")


# Загружаем промпты из файлов
DEFAULT_WEB_SEARCH_PROMPT = _load_prompt_file("web_search.txt")
DEFAULT_RAG_PROMPT = _load_prompt_file("rag.txt")
DEFAULT_COMBINED_PROMPT = _load_prompt_file("combined.txt")


class PromptLoader:
    """Загрузчик промптов для агентов."""

    @staticmethod
    def get_system_prompt(toolkit: list[type], prompts_config: "PromptsConfig") -> str:
        """
        Получить системный промпт для агента.
        Выбирает промпт на основе используемых инструментов:
        - WebSearchTool -> промпт для интернет-поиска
        - RetrievalTool -> промпт для поиска в базе знаний
        - Оба -> комбинированный промпт
        Заменяет {userPost} на информацию о должности пользователя, если она доступна.
        """
        # Если задан кастомный промпт, используем его
        if prompts_config.system_prompt_str:
            system_prompt = prompts_config.system_prompt_str
        else:
            # Определяем, какие инструменты используются
            from api.agents.tools import WebSearchTool, RetrievalTool
            
            has_web_search = any(tool == WebSearchTool or (isinstance(tool, type) and issubclass(tool, WebSearchTool)) for tool in toolkit)
            has_rag = any(tool == RetrievalTool or (isinstance(tool, type) and issubclass(tool, RetrievalTool)) for tool in toolkit)
            
            # Выбираем промпт в зависимости от используемых инструментов
            if has_web_search and has_rag:
                # Комбинация интернет-поиска и базы знаний
                system_prompt = DEFAULT_COMBINED_PROMPT
            elif has_web_search:
                # Только интернет-поиск
                system_prompt = DEFAULT_WEB_SEARCH_PROMPT
            elif has_rag:
                # Только поиск в базе знаний
                system_prompt = DEFAULT_RAG_PROMPT
            else:
                # Дефолтный промпт (если нет ни интернета, ни базы знаний)
                system_prompt = DEFAULT_WEB_SEARCH_PROMPT

        # Заменяем {userPost} на информацию о должности пользователя ПЕРЕД .format()
        # чтобы избежать KeyError при вызове .format() с {available_tools}
        if "{userPost}" in system_prompt:
            user_post = ""
            if prompts_config.user_info and isinstance(prompts_config.user_info, dict):
                user_post = prompts_config.user_info.get("userPost") or ""
            # Замена только {userPost}, без .format(), чтобы не трогать фигурные скобки в JSON
            # Используем тот же формат, что и в simple_llm_call
            if user_post:
                system_prompt = system_prompt.replace("{userPost}", f"Моя должность - {user_post}.")
            else:
                # Если userPost пустой, просто удаляем placeholder
                system_prompt = system_prompt.replace("{userPost}", "")
        
        # Замена {currentDateTime} на текущую дату и время
        if "{currentDateTime}" in system_prompt:
            formatted_date = format_current_datetime()
            system_prompt = system_prompt.replace("{currentDateTime}", formatted_date)
        
        # Добавляем информацию о доступных tools, если есть placeholder
        if "{available_tools}" in system_prompt:
            tool_descriptions = "\n".join([
                f"{i}. {tool.tool_name}: {tool.description or ''}" 
                for i, tool in enumerate(toolkit, start=1)
            ])
            system_prompt = system_prompt.format(available_tools=tool_descriptions)

        return system_prompt

    @staticmethod
    def get_initial_user_request(task_messages: list, prompts_config: "PromptsConfig") -> str:
        """Получить начальный запрос пользователя."""
        if prompts_config.initial_user_request_str:
            return prompts_config.initial_user_request_str
        
        # Извлекаем последнее сообщение пользователя
        if task_messages:
            last_message = task_messages[-1]
            if isinstance(last_message, dict) and last_message.get("role") == "user":
                return last_message.get("content", "")
        
        return "Please help me with my request."

    @staticmethod
    def get_clarification_template(messages: list, prompts_config: "PromptsConfig") -> str:
        """Получить шаблон для запроса уточнений."""
        if prompts_config.clarification_response_str:
            return prompts_config.clarification_response_str
        
        return "The user has provided clarification. Please continue with the task."
