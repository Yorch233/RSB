import tomllib
from pathlib import Path


def test_repository_layout() -> None:
    root = Path(__file__).resolve().parents[1]

    for directory in ("assets", "config", "docs", "RSB", "test"):
        assert (root / directory).is_dir()

    for filename in ("AGENTS.md", "CLAUDE.md", "LICENSE", "README.md", "pyproject.toml"):
        assert (root / filename).is_file()

    for module in (
        "cli/app.py",
        "cli/train/common.py",
        "data/spectral.py",
        "data/datasets/complex_spec.py",
        "metrics/intrusive.py",
        "pipelines/generative.py",
        "workflows/inference.py",
    ):
        assert (root / "RSB" / module).is_file()


def test_deployment_documentation_is_split_out_of_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    for filename in ("installation.md", "datasets.md", "training.md", "inference.md", "metrics.md"):
        assert (root / "docs" / filename).is_file()
        assert f"docs/{filename}" in readme
    assert "**Parameters:**" not in readme


def test_package_and_cli_use_expected_case() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["rsb"] == "RSB.cli.app:app"
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["RSB"]
    lowercase_from = "from " + "rsb"
    lowercase_import = "import " + "rsb"
    for source_path in [*(root / "RSB").rglob("*.py"), *(root / "test").rglob("*.py")]:
        source = source_path.read_text(encoding="utf-8")
        assert lowercase_from not in source
        assert lowercase_import not in source


def test_release_metadata_points_to_public_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    model_source = (root / "RSB" / "modeling_rsb.py").read_text(encoding="utf-8")
    repository = "https://github.com/Yorch233/RSB"

    assert project["version"] == "1.0.0"
    assert project["license"] == {"file": "LICENSE"}
    assert project["urls"]["Repository"] == repository
    assert f'repo_url="{repository}"' in model_source
    assert "gitea-yorch233" not in model_source
    assert "]()" not in readme


def test_claude_guide_references_agent_guide() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
