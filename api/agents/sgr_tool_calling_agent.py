"""
SGR Tool Calling Agent с поддержкой streaming и retry логики.
Адаптировано из sgr-agent-core с добавлением retry логики.
Стриминг опционален (use_streaming в LLMConfig); для GigaChat используется create() без стриминга.
"""

from typing import Literal, Type

import openai
from openai import AsyncOpenAI, pydantic_function_tool
from loguru import logger
from pydantic import ValidationError

from api.agents.agent_definition import AgentConfig
from api.agents.base_agent import BaseAgent
from api.agents.base_tool import BaseTool
from api.agents.tools.reasoning_tool import ReasoningTool


def _format_validation_error(e: ValidationError) -> str:
    """Форматирует Pydantic ValidationError: перечисляет поле и сообщение."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "")
        kind = err.get("type", "")
        parts.append(f"{loc}: {msg} (type={kind})")
    return "; ".join(parts) if parts else str(e)


class SGRToolCallingAgent(BaseAgent):
    """Agent that uses OpenAI native function calling to select and execute
    tools based on SGR like a reasoning scheme."""

    name: str = "sgr_tool_calling_agent"

    def __init__(
        self,
        task_messages: list,
        openai_client: AsyncOpenAI,
        agent_config: AgentConfig,
        toolkit: list[Type[BaseTool]],
        def_name: str | None = None,
        **kwargs: dict,
    ):
        super().__init__(
            task_messages=task_messages,
            openai_client=openai_client,
            agent_config=agent_config,
            toolkit=toolkit,
            def_name=def_name,
            **kwargs,
        )
        self.tool_choice: Literal["required"] = "required"

    def _log_reasoning(self, result: ReasoningTool) -> None:
        """Логирование reasoning фазы."""
        next_step = result.remaining_steps[0] if result.remaining_steps else "Completing"
        sit = (result.current_situation or "")[:400]
        plan = (result.plan_status or "")[:400]
        self.logger.info(
            f"""
    ###############################################
    🤖 LLM RESPONSE DEBUG:
       🧠 Reasoning Steps: {result.reasoning_steps}
       📊 Current Situation: '{sit}...'
       📋 Plan Status: '{plan}...'
       🔍 Searches Done: {self._context.searches_used}
       🔍 Clarifications Done: {self._context.clarifications_used}
       ✅ Enough Data: {result.enough_data}
       📝 Remaining Steps: {result.remaining_steps}
       🏁 Task Completed: {result.task_completed}
       ➡️ Next Step: {next_step}
    ###############################################"""
        )
        self.log.append(
            {
                "step_number": self._context.iteration,
                "timestamp": self.creation_time.isoformat(),
                "step_type": "reasoning",
                "agent_reasoning": result.model_dump(mode="json"),
            }
        )

    async def _reasoning_phase(self) -> ReasoningTool:
        """Call LLM to decide next action based on current context with retry logic."""
        max_retries = self.config.execution.max_retries
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Reasoning phase (попытка {attempt + 1}/{max_retries})")
                kwargs = self.config.llm.to_openai_client_kwargs()
                messages = await self._prepare_context()
                tools_spec = [pydantic_function_tool(ReasoningTool, name=ReasoningTool.tool_name)]
                tool_choice = {"type": "function", "function": {"name": ReasoningTool.tool_name}}

                self.config.llm.use_streaming = True;
                if self.config.llm.use_streaming:
                    async with self.openai_client.chat.completions.stream(
                        messages=messages,
                        tools=tools_spec,
                        tool_choice=tool_choice,
                        **kwargs,
                    ) as stream:
                        async for event in stream:
                            if event.type == "chunk":
                                self.streaming_generator.add_chunk(event.chunk)
                        completion = await stream.get_final_completion()
                    if not completion or not getattr(completion, "choices", None):
                        raise ValueError("Стриминг вернул пустой completion или без choices")
                    if not completion.choices:
                        raise ValueError("completion.choices пустой")
                    msg = completion.choices[0].message if completion.choices[0] else None
                    if not msg or not getattr(msg, "tool_calls", None) or not msg.tool_calls:
                        raise ValueError("Ответ LLM не содержит tool_calls (reasoning)")
                    fn = msg.tool_calls[0].function if msg.tool_calls[0] else None
                    if not fn:
                        raise ValueError("tool_calls[0].function отсутствует")
                    reasoning = getattr(fn, "parsed_arguments", None)
                    if reasoning is None and getattr(fn, "arguments", None):
                        reasoning = ReasoningTool.model_validate_json(fn.arguments)
                    if reasoning is None:
                        raise ValueError("Не удалось получить parsed_arguments для reasoning")
                else:
                    completion = await self.openai_client.chat.completions.create(
                        messages=messages,
                        tools=tools_spec,
                        tool_choice=tool_choice,
                        **kwargs,
                    )
                    msg = completion.choices[0].message
                    if not msg.tool_calls:
                        raise ValueError("LLM не вернул tool call (reasoning)")
                    tc = msg.tool_calls[0]
                    args_str = getattr(tc.function, "arguments", None) or ""
                    reasoning = ReasoningTool.model_validate_json(args_str)
                    if msg.content:
                        self.streaming_generator.add_chunk_from_str(msg.content)

                self.conversation.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"{self._context.iteration}-reasoning",
                                "function": {
                                    "name": reasoning.tool_name,
                                    "arguments": reasoning.model_dump_json(),
                                },
                            }
                        ],
                    }
                )
                tool_call_result = await reasoning(self._context, self.config)
                self.streaming_generator.add_tool_call(
                    f"{self._context.iteration}-reasoning", reasoning.tool_name, tool_call_result
                )
                self.conversation.append(
                    {"role": "tool", "content": tool_call_result, "tool_call_id": f"{self._context.iteration}-reasoning"}
                )
                self._log_reasoning(reasoning)
                return reasoning

            except openai.AuthenticationError as e:
                # Ошибки аутентификации не повторяем
                self.logger.error(f"Ошибка аутентификации LLM API: {e}")
                raise RuntimeError(f"Ошибка аутентификации: {e}") from e
            except openai.RateLimitError as e:
                # Ошибки лимита не повторяем
                self.logger.error(f"Превышен лимит запросов LLM: {e}")
                raise RuntimeError(f"Превышен лимит запросов: {e}") from e
            except openai.BadRequestError as e:
                # Ошибки валидации запроса не повторяем
                self.logger.error(f"Неверный запрос к LLM API: {e}")
                raise RuntimeError(f"Неверный запрос: {e}") from e
            except (openai.APIConnectionError, openai.APITimeoutError, openai.OpenAIError) as e:
                # Ошибки соединения/таймаута/общие ошибки LLM - повторяем
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Ошибка при reasoning (попытка {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    self.logger.error(f"Не удалось выполнить reasoning после {max_retries} попыток: {e}")
                    raise RuntimeError(f"Ошибка reasoning после {max_retries} попыток: {e}") from e
            except ValidationError as e:
                detail = _format_validation_error(e)
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Ошибка валидации reasoning (попытка {attempt + 1}/{max_retries}): {detail}"
                    )
                    continue
                else:
                    self.logger.error(f"Ошибка валидации reasoning после {max_retries} попыток: {detail}")
                    raise RuntimeError(f"Ошибка валидации reasoning: {detail}") from e
            except Exception as e:
                # Другие ошибки - повторяем, если есть попытки; логируем тип и текст
                last_error = e
                err_msg = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"Неожиданная ошибка при reasoning (попытка {attempt + 1}/{max_retries}): {err_msg}"
                    )
                    continue
                else:
                    self.logger.exception("Неожиданная ошибка при reasoning")
                    raise RuntimeError(f"Ошибка reasoning: {err_msg}") from e

        # Если дошли сюда, значит все попытки исчерпаны
        if last_error:
            raise RuntimeError(f"Не удалось выполнить reasoning после {max_retries} попыток: {last_error}") from last_error
        raise RuntimeError("Не удалось выполнить reasoning")

    async def _select_action_phase(self, reasoning: ReasoningTool) -> BaseTool:
        """Select the most suitable tool for the action decided in the reasoning phase with retry logic."""
        max_retries = self.config.execution.max_retries
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Select action phase (попытка {attempt + 1}/{max_retries})")
                kwargs = self.config.llm.to_openai_client_kwargs()
                messages = await self._prepare_context()
                tools_spec = await self._prepare_tools()

                if self.config.llm.use_streaming:
                    async with self.openai_client.chat.completions.stream(
                        messages=messages,
                        tools=tools_spec,
                        tool_choice=self.tool_choice,
                        **kwargs,
                    ) as stream:
                        async for event in stream:
                            if event.type == "chunk":
                                self.streaming_generator.add_chunk(event.chunk)
                        completion = await stream.get_final_completion()
                    tool = completion.choices[0].message.tool_calls[0].function.parsed_arguments
                else:
                    completion = await self.openai_client.chat.completions.create(
                        messages=messages,
                        tools=tools_spec,
                        tool_choice=self.tool_choice,
                        **kwargs,
                    )
                    msg = completion.choices[0].message
                    if not msg.tool_calls:
                        raise ValueError("LLM не вернул tool call (select action)")
                    tc = msg.tool_calls[0]
                    name = getattr(tc.function, "name", None) or ""
                    args_str = getattr(tc.function, "arguments", None) or "{}"
                    tool_class = next((t for t in self.toolkit if getattr(t, "tool_name", None) == name), None)
                    if not tool_class:
                        raise ValueError(f"Неизвестный инструмент: {name}")
                    tool = tool_class.model_validate_json(args_str)
                    if msg.content:
                        self.streaming_generator.add_chunk_from_str(msg.content)

                # Извлекаем tool call из ответа
                if not completion.choices or not completion.choices[0].message.tool_calls:
                    raise ValueError("LLM не вернул tool call. Провайдер может не поддерживать tool calling в streaming режиме.")
                if not isinstance(tool, BaseTool):
                    raise ValueError("Selected tool is not a valid BaseTool instance")
                
                self.conversation.append(
                    {
                        "role": "assistant",
                        "content": reasoning.remaining_steps[0] if reasoning.remaining_steps else "Completing",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"{self._context.iteration}-action",
                                "function": {
                                    "name": tool.tool_name,
                                    "arguments": tool.model_dump_json(),
                                },
                            }
                        ],
                    }
                )
                self.streaming_generator.add_tool_call(
                    f"{self._context.iteration}-action", tool.tool_name, tool.model_dump_json()
                )
                return tool

            except openai.AuthenticationError as e:
                self.logger.error(f"Ошибка аутентификации LLM API: {e}")
                raise RuntimeError(f"Ошибка аутентификации: {e}") from e
            except openai.RateLimitError as e:
                self.logger.error(f"Превышен лимит запросов LLM: {e}")
                raise RuntimeError(f"Превышен лимит запросов: {e}") from e
            except openai.BadRequestError as e:
                self.logger.error(f"Неверный запрос к LLM API: {e}")
                raise RuntimeError(f"Неверный запрос: {e}") from e
            except (openai.APIConnectionError, openai.APITimeoutError, openai.OpenAIError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Ошибка при выборе действия (попытка {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    self.logger.error(f"Не удалось выбрать действие после {max_retries} попыток: {e}")
                    raise RuntimeError(f"Ошибка выбора действия после {max_retries} попыток: {e}") from e
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.logger.warning(f"Неожиданная ошибка при выборе действия (попытка {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    self.logger.exception("Неожиданная ошибка при выборе действия")
                    raise RuntimeError(f"Ошибка выбора действия: {e}") from e

        if last_error:
            raise RuntimeError(f"Не удалось выбрать действие после {max_retries} попыток: {last_error}") from last_error
        raise RuntimeError("Не удалось выбрать действие")

    async def _action_phase(self, tool: BaseTool) -> str:
        """Call Tool for the action decided in the select_action phase."""
        result = await tool(self._context, self.config)
        self.conversation.append(
            {"role": "tool", "content": result, "tool_call_id": f"{self._context.iteration}-action"}
        )
        self.streaming_generator.add_chunk_from_str(f"{result}\n")
        self._log_tool_execution(tool, result)
        return result
