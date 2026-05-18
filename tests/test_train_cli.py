"""Tests CLI : parsing args + chemin --dry-run (qui n'instancie pas de modèle)."""

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from src.train import main, parse_args


def test_parse_args_requires_config():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_basic():
    ns = parse_args(["--config", "configs/tinyllama_r8.yaml"])
    assert ns.config == Path("configs/tinyllama_r8.yaml")
    assert ns.dry_run is False


def test_parse_args_dry_run_flag():
    ns = parse_args(["--config", "x.yaml", "--dry-run"])
    assert ns.dry_run is True


def test_dry_run_prints_config_without_training():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "tinyllama_r8.yaml"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    out = buf.getvalue()
    # le dict sérialisé doit contenir au moins les sections clés
    for key in ("'lora'", "'training'", "'data'", "'quant'"):
        assert key in out
