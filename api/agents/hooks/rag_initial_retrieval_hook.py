"""
Хук: безусловный первый вызов RetrievalTool с точным запросом пользователя до цикла SGR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from api.agents.hooks import BeforeExecutionLoopHook
from api.agents.tools.retrieval_tool import RetrievalTool

if TYPE_CHECKING:
    from api.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RAGInitialRetrievalHook(BeforeExecutionLoopHook):
    """Выполняет первый поиск в базе знаний по точному запросу пользователя до входа в цикл."""

    TOOL_CALL_ID = "0-initial-retrieval"

    async def run(self, agent: BaseAgent) -> None:
        if not self._should_run(agent):
            return
        user_query = self._extract_user_query(agent)
        if not user_query:
            logger.debug("RAGInitialRetrievalHook: не удалось извлечь текст запроса пользователя, пропуск")
            return
        retrieval_tool = RetrievalTool(
            reasoning="Первичный поиск по точному запросу пользователя без перефразирования.",
            query=user_query,
            max_results=5,
        )
        result = await retrieval_tool(agent._context, agent.config)
        agent.conversation.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": self.TOOL_CALL_ID,
                        "type": "function",
                        "function": {
                            "name": retrieval_tool.tool_name,
                            "arguments": retrieval_tool.model_dump_json(),
                        },
                    }
                ],
            }
        )
        agent.conversation.append(
            {"role": "tool", "content": result, "tool_call_id": self.TOOL_CALL_ID}
        )
        if agent.streaming_generator is not None:
            agent.streaming_generator.add_tool_call(
                self.TOOL_CALL_ID, retrieval_tool.tool_name, result
            )
        logger.info("RAGInitialRetrievalHook: выполнен первичный поиск по точному запросу пользователя")

    def _should_run(self, agent: BaseAgent) -> bool:
        has_retrieval = any(
            t is RetrievalTool or (isinstance(t, type) and issubclass(t, RetrievalTool))
            for t in agent.toolkit
        )
        if not has_retrieval:
            return False
        custom = agent._context.custom_context
        if custom is None:
            return False
        if isinstance(custom, dict):
            return bool(custom.get("vdb_url"))
        if hasattr(custom, "model_dump"):
            d = getattr(custom, "model_dump", lambda: {})()
            return bool(isinstance(d, dict) and d.get("vdb_url"))
        return False

    def _extract_user_query(self, agent: BaseAgent) -> str | None:
        for msg in agent.task_messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                return content.strip() or None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            return text.strip()
                        break
        return None
