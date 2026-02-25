"""
Хуки, выполняемые до основного цикла агента (например, безусловный первый RAG-поиск).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.agents.base_agent import BaseAgent


class BeforeExecutionLoopHook(ABC):
    """Протокол хука, вызываемого один раз перед входом в цикл execute()."""

    @abstractmethod
    async def run(self, agent: "BaseAgent") -> None:
        """Выполнить логику до основного цикла (например, первый RetrievalTool с точным запросом)."""
        ...
