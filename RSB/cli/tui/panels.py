"""Theme-consistent Rich panel and table factories."""

from typing import Any

from rich.panel import Panel
from rich.table import Table

from RSB.cli.tui.theme import ACCENT, BG_INNER, TEXT


def make_panel(
    content: Any,
    *,
    title: str | None = None,
    border: str = ACCENT,
    padding: tuple[int, int] = (0, 1),
) -> Panel:
    """Create a panel using the shared RSB visual language."""
    styled_title = f"[bold {border}]{title}[/]" if title else None
    return Panel(content, title=styled_title, border_style=border, padding=padding, style=f"on {BG_INNER}")


def make_kv_panel(
    rows: list[tuple[str, str]],
    *,
    title: str | None = None,
    border: str = ACCENT,
    key_style: str = f"bold {TEXT}",
    value_style: str = TEXT,
) -> Panel:
    """Create a key-value table inside a themed panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style=key_style, no_wrap=True)
    table.add_column(style=value_style)
    for key, value in rows:
        table.add_row(key, value)
    return make_panel(table, title=title, border=border)
