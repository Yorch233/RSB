from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from RSB.cli.tui import menu, prompts


class _FakeTerminalConsole:
    def __init__(self, answers: list[str]) -> None:
        self.file = StringIO()
        self.is_terminal = True
        self._answers = iter(answers)

    def print(self, value: str = "", *args, **kwargs) -> None:
        del args, kwargs
        self.file.write(f"{value}\n")

    def input(self, prompt: str) -> str:
        self.file.write(prompt)
        return next(self._answers)


def test_escape_key_does_not_wait_for_an_escape_sequence(monkeypatch) -> None:
    characters = iter(["\x1b", None])
    monkeypatch.setattr(menu, "_read_character", lambda timeout=None: next(characters))

    assert menu._read_key() == "escape"


def test_fallback_menu_converts_eof_to_clean_cancellation(monkeypatch) -> None:
    class Console:
        def print(self, *args, **kwargs) -> None:
            del args, kwargs

        def input(self, prompt: str) -> str:
            del prompt
            raise EOFError

    monkeypatch.setattr(menu, "get_console", Console)

    with pytest.raises(SystemExit, match="1"):
        menu.SelectionMenu("Choose", ["one", "two"])._fallback(0)


def test_selection_menu_is_vertical_left_aligned_and_unboxed() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    console.print(menu.SelectionMenu("Mixed precision", ["none", "fp16", "bf16"])._render(1))

    rendered = output.getvalue()
    assert rendered.startswith(
        "◇ Mixed precision?\n"
        "│  1. ○ none\n"
        "│  2. ● fp16\n"
        "│  3. ○ bf16\n"
        "│  Use ↑/↓ or a number, then press Enter. Default: fp16"
    )
    assert "╭" not in rendered
    assert "─" not in rendered

    output.seek(0)
    output.truncate()
    console.print(menu.SelectionMenu("Logger", ["wandb", "none"])._render(0, show_hint=False))
    assert output.getvalue().startswith("◇ Logger?\n│  1. ● wandb\n│  2. ○ none")


def test_text_prompt_collapses_to_question_and_final_answer(monkeypatch) -> None:
    console = _FakeTerminalConsole(["custom-runs"])
    monkeypatch.setattr(prompts, "get_console", lambda: console)

    result = prompts.prompt_text(
        "Run directory",
        default="runs",
        description="Where training runs are stored.",
        input_hint="Enter a path.",
    )

    assert result == "custom-runs"
    rendered = console.file.getvalue()
    visible = rendered.rsplit("\x1b[J", maxsplit=1)[1]
    assert rendered.count("\x1b[s") == 1
    assert "Run directory?" in visible
    assert "Answer:" in visible
    assert "custom-runs" in visible
    assert "Where training runs are stored." not in visible
    assert "Default:" not in visible
    assert "Enter a path." not in visible


def test_validated_text_redraws_errors_and_collapses_success(monkeypatch) -> None:
    console = _FakeTerminalConsole(["bad", "valid"])
    monkeypatch.setattr(prompts, "get_console", lambda: console)

    def validate(value: str) -> str:
        if value == "bad":
            raise ValueError("Invalid value.")
        return value

    result = prompts.prompt_validated_text(
        "Dataset ID",
        validate,
        description="Enter a unique dataset ID.",
        input_hint="Letters and numbers only.",
    )

    assert result == "valid"
    rendered = console.file.getvalue()
    visible = rendered.rsplit("\x1b[J", maxsplit=1)[1]
    assert rendered.count("\x1b[s") == 1
    assert rendered.count("\x1b[u\x1b[J") == 2
    assert "Dataset ID?" in visible
    assert "Answer:" in visible
    assert "valid" in visible
    assert "bad" not in visible
    assert "Invalid value." not in visible
    assert "Enter a unique dataset ID." not in visible
    assert "Letters and numbers only." not in visible
