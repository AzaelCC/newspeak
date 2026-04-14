from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from newspeak.adapters.llm import OpenAIChatClient
from newspeak.config import CoachConfig

if TYPE_CHECKING:
    from newspeak.services.modes import Mode

logger = logging.getLogger(__name__)

NO_NOTE_PHRASES = {"no note", "no note.", "no note!"}


@dataclass
class CoachNote:
    turn_id: str
    mode_id: str
    note: str
    skipped: bool
    deep: bool


class CoachService:
    def __init__(self, client: OpenAIChatClient, config: CoachConfig) -> None:
        self._client = client
        self._config = config

    async def analyze(
        self,
        *,
        turn_id: str,
        transcription: str,
        assistant_text: str,
        roleplay_history: list[dict[str, Any]],
        prior_notes: list[CoachNote],
        mode: Mode,
        deep: bool,
    ) -> CoachNote:
        system_prompt = mode.coach_system_deep if deep else mode.coach_system_auto
        window = self._config.history_window * 2

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # Rolling window of roleplay history for pragmatic context (read-only copy)
        messages.extend(roleplay_history[-window:])

        # Prior coach notes so the coach doesn't repeat itself
        meaningful_prior = [n for n in prior_notes if not n.skipped and n.note]
        if meaningful_prior:
            prior_text = "\n- ".join(n.note for n in meaningful_prior)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Previous notes you already gave (avoid repeating):\n- {prior_text}"
                    ),
                }
            )

        user_content = (
            f"The learner just said: {transcription!r}\nThe character replied: {assistant_text!r}\n"
        )
        if deep:
            user_content += (
                "Provide a thorough deep-dive critique covering structure, alternatives, "
                "and patterns."
            )
        else:
            user_content += (
                "Give one short note, or reply with exactly 'no note' if there is nothing "
                "meaningful to say."
            )

        messages.append({"role": "user", "content": user_content})

        model = self._config.model or "default"
        max_tokens = self._config.deep_max_tokens if deep else self._config.auto_max_tokens

        text = await self._client.complete_messages(
            model=model, messages=messages, max_tokens=max_tokens
        )

        skipped = (not deep) and text.lower().strip() in NO_NOTE_PHRASES
        return CoachNote(
            turn_id=turn_id,
            mode_id=mode.id,
            note="" if skipped else text,
            skipped=skipped,
            deep=deep,
        )
