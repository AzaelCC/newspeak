from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Mode:
    id: str
    name: str
    roleplay_system: str
    coach_system_auto: str
    coach_system_deep: str
    coach_language: str = "English"
    target_language: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "coach_language": self.coach_language,
            "target_language": self.target_language,
        }


class ModeRegistry:
    def __init__(self) -> None:
        self._modes: dict[str, Mode] = {}

    def register(self, mode: Mode) -> None:
        self._modes[mode.id] = mode

    def get(self, mode_id: str) -> Mode | None:
        return self._modes.get(mode_id)

    def list(self) -> list[Mode]:
        return list(self._modes.values())

    def list_dicts(self) -> list[dict]:
        return [m.to_dict() for m in self._modes.values()]
