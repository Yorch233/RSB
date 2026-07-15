"""Shared Rich console for RSB terminal output."""

from rich.console import Console

from RSB.cli.tui.theme import RSB_THEME

_CONSOLE = Console(theme=RSB_THEME)


def get_console() -> Console:
    """Return the process-wide themed console."""
    return _CONSOLE
