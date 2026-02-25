"""
Базовый класс для агентов.
Адаптировано из sgr-agent-core.
"""

import importlib
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Type

from openai import AsyncOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam

from api.agents.agent_definition import AgentConfig
from api.agents.models import AgentContext, AgentStatesEnum
from api.agents.services.prompt_loader import PromptLoader
from api.agents.registry import AgentRegistry
from api.agents.stream import OpenAIStreamingGenerator
from api.agents.base_tool import BaseTool
from loguru import logger


class AgentRegistryMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ not in ("BaseAgent",):
            AgentRegistry.register(cls, name=cls.name)


class BaseAgent(AgentRegistryMixin):
    """Base class for agents."""

    name: str = "base_agent"

    def __init__(
        self,
        task_messages: list[ChatCompletionMessageParam],
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        self.id = f"{def_name or self.name}_{uuid.uuid4()}"
        self.openai_client = openai_client
        self.config = agent_config
        self.creation_time = datetime.now()
        self.task_messages = task_messages
        self.toolkit = toolkit

        self._context = AgentContext()
        self.conversation = []

        self.streaming_generator = OpenAIStreamingGenerator(model=self.config.llm.model or "gpt-4o")
        self.logger = logger.bind(agent_id=self.id)
        self.log = []

    async def provide_clarification(self, messages: list[ChatCompletionMessageParam]):
        """Receive clarification from an external source (e.g. user input) in
        OpenAI messages format."""
        self.conversation.extend(messages)
        self.conversation.append(
            {"role": "user", "content": PromptLoader.get_clarification_template(messages, self.config.prompts)}
        )

        self._context.clarifications_used += 1
        self._context.clarification_received.set()
        self._context.state = AgentStatesEnum.RESEARCHING
        self.logger.info(f"✅ Clarification received: {len(messages)} messages")

    def _log_reasoning(self, result) -> None:
        """Логирование reasoning фазы."""
        # Будет переопределено в подклассах с конкретным типом ReasoningTool
        self.logger.debug(f"Reasoning phase completed: {result}")

    def _log_tool_execution(self, tool: BaseTool, result: str):
        """Логирование выполнения tool."""
        self.logger.info(
            f"""
###############################################
🛠️ TOOL EXECUTION DEBUG:
    🔧 Tool Name: {tool.tool_name}
    📋 Tool Model: {tool.model_dump_json(indent=2)}
    🔍 Result: '{result[:400]}...'
###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": datetime.now().isoformat(),
                "step_type": "tool_execution",
                "tool_name": tool.tool_name,
                "agent_tool_context": tool.model_dump(mode="json"),
                "agent_tool_execution_result": result,
            }
        )

    def _save_agent_log(self):
        """Сохранение логов агента."""
        logs_dir = self.config.execution.logs_dir
        # Skip saving if logs_dir is None or empty string
        if not logs_dir:
            self.logger.debug("Skipping agent log save: logs_dir is not configured")
            return

        os.makedirs(logs_dir, exist_ok=True)
        filepath = os.path.join(logs_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.id}-log.json")
        agent_log = {
            "id": self.id,
            "model_config": self.config.llm.model_dump(
                exclude={"api_key", "proxy"}, mode="json"
            ),  # Sensitive data excluded by default
            "task_messages": self.task_messages,
            "toolkit": [tool.tool_name for tool in self.toolkit],
            "log": self.log,
        }

        json.dump(agent_log, open(filepath, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    async def _prepare_context(self) -> list[dict]:
        """Prepare a conversation context with system prompt, task data and any
        other context.

        Note: Override this method to change the context setup for the agent.

        Returns a list of dictionaries OpenAI like format, each
        containing a role and content key by default.
        """

        # Формируем контекст: системный промпт + task_messages + conversation
        context_messages = [
            {"role": "system", "content": PromptLoader.get_system_prompt(self.toolkit, self.config.prompts)},
        ]
        
        # Добавляем task_messages (они уже содержат запрос пользователя)
        context_messages.extend(self.task_messages)
        
        # Добавляем conversation (история взаимодействия агента с tools)
        context_messages.extend(self.conversation)
        
        return context_messages

    async def _prepare_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Prepare available tools for the current agent state and progress.

        Note: Override this method to change the tool setup or conditions for tool
        usage.

        Returns a list of ChatCompletionFunctionToolParam based
        available tools.
        """
        tools = set(self.toolkit)
        if self._context.iteration >= self.config.execution.max_iterations:
            raise RuntimeError("Max iterations reached")
        return [pydantic_function_tool(tool, name=tool.tool_name) for tool in tools]

    async def _reasoning_phase(self):
        """Call LLM to decide next action based on current context."""
        raise NotImplementedError("_reasoning_phase must be implemented by subclass")

    async def _select_action_phase(self, reasoning):
        """Select the most suitable tool for the action decided in the
        reasoning phase.

        Returns the tool suitable for the action.
        """
        raise NotImplementedError("_select_action_phase must be implemented by subclass")

    async def _action_phase(self, tool: BaseTool) -> str:
        """Call Tool for the action decided in the select_action phase.

        Returns string or dumped JSON result of the tool execution.
        """
        raise NotImplementedError("_action_phase must be implemented by subclass")

    async def _execution_step(self):
        """Execute a single step of the agent workflow.

        Note: Override this method to change the agent workflow for each step.
        """
        reasoning = await self._reasoning_phase()
        self._context.current_step_reasoning = reasoning
        action_tool = await self._select_action_phase(reasoning)
        await self._action_phase(action_tool)

        # ClarificationTool будет обработан в подклассах при необходимости
        # if isinstance(action_tool, ClarificationTool):
        #     self.logger.info("\n⏸️  Research paused - please answer questions")
        #     self._context.state = AgentStatesEnum.WAITING_FOR_CLARIFICATION
        #     self.streaming_generator.finish()
        #     self._context.clarification_received.clear()
        #     await self._context.clarification_received.wait()

    async def execute(
        self,
    ):
        self.logger.info(f"🚀 User provided {len(self.task_messages)} messages.")
        try:
            hook_spec = getattr(self.config, "before_execution_loop_hook", None)
            if hook_spec is not None:
                hook_class = hook_spec
                if isinstance(hook_spec, str):
                    module_path, _, class_name = hook_spec.rpartition(".")
                    if module_path and class_name:
                        module = importlib.import_module(module_path)
                        hook_class = getattr(module, class_name)
                if hook_class is not None:
                    hook = hook_class()
                    await hook.run(self)
            while self._context.state not in AgentStatesEnum.FINISH_STATES.value:
                self._context.iteration += 1
                self.logger.info(f"Step {self._context.iteration} started")
                await self._execution_step()
            return self._context.execution_result

        except Exception as e:
            self.logger.error(f"❌ Agent execution error: {str(e)}")
            self._context.state = AgentStatesEnum.FAILED
            self._context.execution_result = str(e)
            traceback.print_exc()
        finally:
            if self.streaming_generator is not None:
                self.streaming_generator.finish(self._context.execution_result)
            self._save_agent_log()
