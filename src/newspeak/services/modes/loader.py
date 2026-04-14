from __future__ import annotations

import logging
from pathlib import Path

from newspeak.services.modes import Mode, ModeRegistry

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"id", "name", "roleplay_system", "coach_system_auto", "coach_system_deep"}


def load_custom_modes(registry: ModeRegistry, custom_modes_dir: Path) -> int:
    """Scan custom_modes_dir for *.yaml files and register valid modes. Returns count loaded."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — custom modes will not be loaded")
        return 0

    if not custom_modes_dir.exists():
        return 0

    loaded = 0
    for path in sorted(custom_modes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("Custom mode %s: expected a YAML mapping, skipping", path.name)
                continue

            missing = REQUIRED_KEYS - data.keys()
            if missing:
                logger.warning(
                    "Custom mode %s: missing required keys %s, skipping", path.name, missing
                )
                continue

            target_language = str(data["target_language"]) if data.get("target_language") else None
            mode = Mode(
                id=str(data["id"]),
                name=str(data["name"]),
                description=str(data.get("description", "")),
                roleplay_system=str(data["roleplay_system"]),
                coach_system_auto=str(data["coach_system_auto"]),
                coach_system_deep=str(data["coach_system_deep"]),
                coach_language=str(data.get("coach_language", "English")),
                target_language=target_language,
            )
            registry.register(mode)
            loaded += 1
            logger.info("Loaded custom mode %r from %s", mode.id, path.name)
        except Exception:
            logger.exception("Failed to load custom mode from %s", path.name)

    return loaded
