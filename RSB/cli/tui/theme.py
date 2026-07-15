"""Centralized theme tokens for the RSB console interface."""

from rich.theme import Theme

TEXT = "#f5f5f5"
DIM = "#888888"
MUTED = "#555555"
ACCENT = "#f97316"
ACCENT_LIGHT = "#fb923c"
BG = "#0a0a0a"
BG_CARD = "#111111"
BG_INNER = "#0d0d0d"
SUCCESS = "#4ade80"
ERROR = "#f87171"
INFO = "#60a5fa"

PHASE_PENDING = "○"
PHASE_RUNNING = "◐"
PHASE_DONE = "✓"

PROGRESS_FILLED = "█"
PROGRESS_EMPTY = "░"

RSB_THEME = Theme(
    {
        "repr.str": TEXT,
        "text": TEXT,
        "dim": DIM,
        "muted": MUTED,
        "accent": ACCENT,
        "accent.light": ACCENT_LIGHT,
        "success": SUCCESS,
        "error": ERROR,
        "info": INFO,
        "black": "black",
        "red": "#ef4444",
        "green": "#22c55e",
        "orange": ACCENT,
        "blue": "#3b82f6",
        "purple": "#a855f7",
        "cyan": "#06b6d7",
        "white": TEXT,
        "bright_black": MUTED,
        "bright_red": ERROR,
        "bright_green": SUCCESS,
        "bright_orange": ACCENT_LIGHT,
        "bright_blue": INFO,
        "bright_purple": "#c084fc",
        "bright_cyan": "#22d3ee",
        "bright_white": "#ffffff",
    }
)
