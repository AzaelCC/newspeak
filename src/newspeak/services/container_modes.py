from newspeak.config import CoachConfig
from newspeak.services.modes import ModeRegistry
from newspeak.services.modes.builtin import register_builtin_modes
from newspeak.services.modes.loader import load_custom_modes


def build_mode_registry(config: CoachConfig) -> ModeRegistry:
    registry = ModeRegistry()
    register_builtin_modes(registry)
    load_custom_modes(registry, config.custom_modes_dir)
    return registry
