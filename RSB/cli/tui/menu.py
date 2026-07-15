"""Arrow-key terminal selection menu used by RSB commands.

The interaction model is based on Hugging Face Accelerate's terminal menu:
https://github.com/huggingface/accelerate/tree/main/src/accelerate/commands/menu/
"""

from __future__ import annotations

import os
import select
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.text import Text

from RSB.cli.tui.console import get_console


def as_question(prompt: str) -> str:
    """Normalize an interactive prompt as a question."""
    normalized = prompt.strip()
    return normalized if normalized.endswith(("?", "？")) else f"{normalized}?"


@contextmanager
def _raw_terminal() -> Iterator[None]:
    """Temporarily put a POSIX terminal into cbreak input mode."""
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        # Keep output processing enabled so newlines remain aligned at column
        # zero while still receiving arrow keys without waiting for Enter.
        tty.setcbreak(descriptor)
        yield
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _read_character(timeout: float | None = None) -> str | None:
    descriptor = sys.stdin.fileno()
    if timeout is not None:
        readable, _, _ = select.select([descriptor], [], [], timeout)
        if not readable:
            return None
    character = os.read(descriptor, 1)
    if not character:
        raise EOFError
    return character.decode(errors="ignore")


def _read_key() -> str:
    """Read one key, translating common ANSI arrow sequences."""
    character = _read_character()
    if character == "\x03":
        raise KeyboardInterrupt
    if character != "\x1b":
        assert character is not None
        return character
    prefix = _read_character(0.05)
    if prefix != "[":
        return "escape"
    suffix = _read_character(0.05)
    return {"A": "up", "B": "down"}.get(suffix or "", "escape")


class SelectionMenu:
    """Render a keyboard-driven, single-choice terminal menu."""

    def __init__(self, prompt: str, choices: list[str]) -> None:
        """Initialize a menu with a prompt and at least one choice."""
        if not choices:
            raise ValueError("SelectionMenu requires at least one choice")
        self.prompt = as_question(prompt)
        self.choices = choices

    def _render(self, position: int, *, show_hint: bool = True, default: int | None = None) -> Text:
        rendered = Text.assemble(("◇ ", "accent"), (self.prompt, "bold text"))
        for index, choice in enumerate(self.choices):
            rendered.append("\n│  ", style="dim")
            rendered.append(f"{index + 1}. ", style="dim")
            if index == position:
                rendered.append("● ", style="accent")
                rendered.append(choice, style="bold accent")
            else:
                rendered.append("○ ", style="dim")
                rendered.append(choice, style="text")
        if show_hint:
            default_position = position if default is None else default
            rendered.append(
                f"\n│  Use ↑/↓ or a number, then press Enter. Default: {self.choices[default_position]}",
                style="dim",
            )
        elif default is not None:
            rendered.append(f"\n│  Enter an option number. Default: {self.choices[default]}", style="dim")
        return rendered

    def _print_result(self, position: int) -> None:
        result = Text.assemble(
            ("◇ ", "success"),
            (self.prompt, "text"),
            ("\n│  Selected: ", "dim"),
            (self.choices[position], "accent"),
            ("\n│", "dim"),
        )
        get_console().print(result)

    def _fallback(self, default: int) -> int:
        """Use numbered line input when raw terminal control is unavailable."""
        console = get_console()
        console.print(self._render(default, show_hint=False, default=default))
        while True:
            try:
                value = console.input(
                    f"[dim]│  Enter a number from 1 to {len(self.choices)} (default: {default + 1}): [/dim]"
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]│[/dim] [error]Configuration cancelled.[/error]")
                raise SystemExit(1) from None
            if not sys.stdin.isatty():
                console.print()
                console.print("[dim]│[/dim]")
            if not value:
                return default
            if value.isdigit() and 1 <= int(value) <= len(self.choices):
                return int(value) - 1
            console.print(f"[dim]│[/dim] [error]Enter a number from 1 to {len(self.choices)}.[/error]")

    def run(self, default: int = 0) -> int:
        """Return the selected zero-based choice index."""
        if not 0 <= default < len(self.choices):
            raise ValueError("default selection is outside the choice range")
        if os.name != "posix" or not sys.stdin.isatty():
            position = self._fallback(default)
            self._print_result(position)
            return position

        from rich.live import Live

        console = get_console()
        position = default
        with _raw_terminal(), Live(console=console, transient=True, auto_refresh=False) as live:
            while True:
                live.update(self._render(position, default=default), refresh=True)
                key = _read_key()
                if key == "up":
                    position = (position - 1) % len(self.choices)
                elif key == "down":
                    position = (position + 1) % len(self.choices)
                elif key in {"\r", "\n"}:
                    break
                elif key == "escape":
                    raise KeyboardInterrupt
                elif key.isdigit() and 1 <= int(key) <= len(self.choices):
                    position = int(key) - 1
        self._print_result(position)
        return position
