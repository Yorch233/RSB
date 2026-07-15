"""Canonical project paths used by command-line workflows."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yml"
USER_CONFIG_PATH = PROJECT_ROOT / ".config" / "rsb.yml"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

DNSMOS_DIR = PROJECT_ROOT / "RSB" / "metrics" / "dnsmos"
DNSMOS_MODEL_DIR = DNSMOS_DIR / "models"
DNSMOS_PERSONALIZED_MODEL_DIR = DNSMOS_DIR / "personalized_models"
