"""Custom help display building blocks for the RSB CLI."""

from click import Context, HelpFormatter
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typer.core import TyperGroup

from RSB.cli.tui.console import get_console
from RSB.cli.tui.theme import ACCENT, BG_INNER, DIM, MUTED


def _build_section(title: str, items: list[tuple[str, str]]) -> Group:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style=f"bold {ACCENT}", no_wrap=True, width=20)
    table.add_column(style=DIM, ratio=1)
    for key, description in items:
        table.add_row(key, description)
    return Group(Text(title, style=f"bold {ACCENT}"), table)


def print_help(commands: list[tuple[str, str]], options: list[tuple[str, str]]) -> None:
    """Print an RSB-branded command overview panel."""
    usage = Text("  rsb ", style=DIM)
    usage.append("<command>", style=ACCENT)
    usage.append(" [options]", style=DIM)
    footer = Text("Run ", style=DIM)
    footer.append("rsb <command> --help", style=f"bold {ACCENT}")
    footer.append(" for detailed usage.", style=DIM)
    content = Group(
        Text("Usage:", style=f"bold {ACCENT}"),
        usage,
        Text(),
        _build_section("Commands:", commands),
        Text(),
        _build_section("Options:", options),
        Text(),
        Rule(style=MUTED),
        footer,
    )
    get_console().print(
        Panel(
            content,
            title=f"[dim]rsb --help[/]  [{ACCENT}]RSB CLI[/]",
            border_style="#1a1a1a",
            style=f"on {BG_INNER}",
            padding=(1, 2),
            box=box.ROUNDED,
        )
    )


class ThemedTyperGroup(TyperGroup):
    """Render the root Typer help page with the shared RSB theme."""

    def format_help(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Print themed command and option sections instead of Click's formatter output."""
        del formatter
        commands: list[tuple[str, str]] = []
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is not None and not command.hidden:
                commands.append((name, command.get_short_help_str(limit=1000)))

        options: list[tuple[str, str]] = []
        for parameter in self.get_params(ctx):
            record = parameter.get_help_record(ctx)
            if record is not None:
                options.append(record)
        print_help(commands, options)
