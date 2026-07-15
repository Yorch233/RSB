"""Configuration, help, and status display components."""

from __future__ import annotations

import enum
from typing import Any

import yaml
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from RSB.cli.tui.console import get_console
from RSB.cli.tui.panels import make_kv_panel, make_panel
from RSB.cli.tui.theme import ACCENT, DIM, MUTED, PHASE_DONE, PHASE_PENDING, PHASE_RUNNING, SUCCESS, TEXT


class StatusPhase(enum.Enum):
    """Task lifecycle phase."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


_PHASE_SYMBOLS: dict[StatusPhase, tuple[str, str]] = {
    StatusPhase.PENDING: (PHASE_PENDING, MUTED),
    StatusPhase.RUNNING: (PHASE_RUNNING, ACCENT),
    StatusPhase.DONE: (PHASE_DONE, SUCCESS),
}


def print_config_summary(
    config: dict[str, Any],
    *,
    title: str = "Configuration",
    show_yaml: bool = True,
    boxed: bool = True,
) -> None:
    """Print configuration as themed YAML or a flattened key-value panel."""
    if show_yaml:
        yaml_text = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_text, "yaml", theme="ansi_dark")
        if boxed:
            get_console().print(make_panel(syntax, title=title))
        else:
            get_console().print(f"[accent]◇[/accent] [bold]{title}[/bold]")
            guide = Text("\n".join("│" for _ in yaml_text.splitlines()), style="dim")
            table = Table.grid(padding=(0, 1))
            table.add_column(width=1)
            table.add_column()
            table.add_row(guide, syntax)
            get_console().print(table)
            get_console().print("[dim]│[/dim]")
        return

    def flatten(mapping: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for key, value in mapping.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                rows.extend(flatten(value, full_key))
            else:
                rows.append((full_key, str(value)))
        return rows

    rows = flatten(config)
    if boxed:
        get_console().print(make_kv_panel(rows, title=title))
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", width=1)
    table.add_column(style=f"bold {TEXT}", no_wrap=True)
    table.add_column(style=TEXT)
    for key, value in rows:
        table.add_row("│", key, value)
    get_console().print(f"[accent]◇[/accent] [bold]{title}[/bold]")
    get_console().print(table)
    get_console().print("[dim]│[/dim]")


def print_help_table(
    title: str,
    items: list[tuple[str, str]],
    *,
    key_header: str = "Option",
    value_header: str = "Description",
) -> None:
    """Print a help-style table inside a themed panel."""
    table = Table(show_header=True, header_style=f"bold {ACCENT}", padding=(0, 1))
    table.add_column(key_header, style=f"bold {TEXT}", no_wrap=True)
    table.add_column(value_header, style=DIM)
    for key, description in items:
        table.add_row(key, description)
    get_console().print(make_panel(table, title=title))


def render_phase(label: str, phase: StatusPhase) -> str:
    """Render a status phase using the shared glyph and color tokens."""
    symbol, color = _PHASE_SYMBOLS[phase]
    return f"[{color}]{symbol}[/] {label}"


def print_phase(label: str, phase: StatusPhase) -> None:
    """Print a themed status phase."""
    get_console().print(render_phase(label, phase))
