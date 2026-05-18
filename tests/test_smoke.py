"""Smoke test — structure de base du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_layout():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "src" / "__init__.py").is_file()
    assert (ROOT / "tests").is_dir()
    assert (ROOT / "configs").is_dir()
    assert (ROOT / "runs").is_dir()
