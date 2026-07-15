"""Rich-styled input, selection, and confirmation prompts."""

from __future__ import annotations

import sys
from collections.abc import Callable

from rich.markup import escape

from RSB.cli.tui.console import get_console
from RSB.cli.tui.menu import SelectionMenu, as_question
from RSB.cli.tui.theme import ACCENT, ERROR


class _PromptCheckpoint:
    """Restore a terminal prompt to its initial cursor position."""

    def __init__(self) -> None:
        self.console = get_console()
        self.enabled = self.console.is_terminal
        if self.enabled:
            self.console.file.write("\x1b[s")
            self.console.file.flush()

    def rollback(self) -> bool:
        """Erase the current prompt block when terminal control is available."""
        if not self.enabled:
            return False
        self.console.file.write("\x1b[u\x1b[J")
        self.console.file.flush()
        return True


def _print_answer(label: str, result: str) -> None:
    console = get_console()
    console.print(f"[{ACCENT}]◇[/] [bold]{escape(as_question(label))}[/]")
    console.print(f"[dim]│  Answer:[/dim] [{ACCENT}]{escape(result)}[/]")
    console.print("[dim]│[/dim]")


def prompt_text(
    label: str,
    default: str | None = None,
    description: str | None = None,
    input_hint: str | None = None,
    error: str | None = None,
    *,
    collapse: bool = True,
) -> str:
    """Prompt for text input, then collapse it to the question and answer."""
    checkpoint = _PromptCheckpoint() if collapse else None
    console = get_console()
    console.print(f"[{ACCENT}]◇[/] [bold]{escape(as_question(label))}[/]")
    if description:
        console.print(f"[dim]│  {escape(description)}[/dim]")
    if default is not None:
        console.print(f"[dim]│  Default:[/dim] [{ACCENT}]{escape(default)}[/]")
    if error:
        console.print(f"[{ERROR}]│  {escape(error)}[/]")
    hint = input_hint or (
        "Enter a value, or press Enter to use the default." if default is not None else "Enter a value."
    )
    console.print(f"[dim]│  {escape(hint)}[/dim]")
    try:
        value = console.input(f"[bold {ACCENT}]│  > [/]").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]│[/dim] [error]Configuration cancelled.[/error]")
        raise SystemExit(1) from None
    if not sys.stdin.isatty():
        console.print()
    result = value if value else (default or "")
    if checkpoint is not None and checkpoint.rollback():
        _print_answer(label, result)
    else:
        console.print(f"[dim]│  Answer:[/dim] [{ACCENT}]{escape(result)}[/]")
        console.print("[dim]│[/dim]")
    return result


def prompt_validated_text[ValidatedValue](
    label: str,
    validator: Callable[[str], ValidatedValue],
    *,
    default: str | None = None,
    description: str | None = None,
    input_hint: str | None = None,
) -> ValidatedValue:
    """Prompt until validation succeeds, redrawing failed input in place."""
    checkpoint = _PromptCheckpoint()
    error: str | None = None
    while True:
        value = prompt_text(
            label,
            default,
            description,
            input_hint,
            error,
            collapse=False,
        )
        try:
            validated = validator(value)
        except ValueError as validation_error:
            checkpoint.rollback()
            error = str(validation_error)
            continue
        if checkpoint.rollback():
            _print_answer(label, value)
        return validated


def prompt_number(
    label: str,
    *,
    default: int | float | None = None,
    num_type: type[int] | type[float] = int,
    input_hint: str | None = None,
) -> int | float:
    """Prompt until the user enters a valid numeric value."""
    type_name = "integer" if num_type is int else "number"
    hint = input_hint or f"Enter a valid {type_name}, or press Enter to use the default."

    def validate(value: str) -> int | float:
        """Convert the entered value to the requested numeric type."""
        try:
            return num_type(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Please enter a valid {type_name}.") from error

    return prompt_validated_text(
        label,
        validate,
        default=str(default) if default is not None else None,
        input_hint=hint,
    )


def rich_select(prompt: str, choices: list[str], *, default: int = 0) -> int:
    """Display a styled selection list and return its selected index."""
    return SelectionMenu(prompt, choices).run(default)


def prompt_yes_no(question: str, *, default: bool = False) -> bool:
    """Prompt for yes/no with the arrow-key selection menu."""
    return rich_select(question, ["Yes", "No"], default=0 if default else 1) == 0
