import json
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from newspeak.services.ports import ChatResult
from newspeak.services.prompts import TRANSCRIPTION_TOOLS, system_prompt_for


class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str = "not-needed"):
        self._client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)

    async def complete(
        self,
        *,
        model: str,
        history: Sequence[dict[str, Any]],
        user_message: dict[str, Any],
        use_transcription_tool: bool,
        roleplay_system: str | None = None,
    ) -> ChatResult:
        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_for(use_transcription_tool, roleplay_system),
                },
                *history,
                user_message,
            ],
        }
        if use_transcription_tool:
            completion_kwargs.update(
                {
                    "tools": TRANSCRIPTION_TOOLS,
                    "tool_choice": "required",
                }
            )

        completion = await self._client.chat.completions.create(**completion_kwargs)
        choice = completion.choices[0].message

        if use_transcription_tool and choice.tool_calls:
            args = json.loads(choice.tool_calls[0].function.arguments)
            return ChatResult(
                text=args.get("response", "").strip(),
                transcription=args.get("transcription", "").strip() or None,
                mode="tool",
            )

        return ChatResult(text=(choice.content or "").strip(), mode="chat")

    async def complete_messages(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
    ) -> str:
        """Send an arbitrary list of messages and return the assistant text."""
        kwargs: dict = {"model": model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        completion = await self._client.chat.completions.create(**kwargs)
        return (completion.choices[0].message.content or "").strip()
