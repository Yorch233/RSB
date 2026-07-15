from RSB.utils.paths import (
    ASSETS_DIR,
    CONFIG_DIR,
    DNSMOS_MODEL_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    USER_CONFIG_PATH,
)


def test_project_paths_resolve_inside_repository() -> None:
    assert ASSETS_DIR == PROJECT_ROOT / "assets"
    assert CONFIG_DIR == PROJECT_ROOT / "config"
    assert USER_CONFIG_PATH == PROJECT_ROOT / ".config" / "rsb.yml"
    assert RUNS_DIR == PROJECT_ROOT / "runs"


def test_dnsmos_assets_exist() -> None:
    assert (DNSMOS_MODEL_DIR / "model_v8.onnx").is_file()
    assert (DNSMOS_MODEL_DIR / "sig_bak_ovr.onnx").is_file()
