"""Tests du parsing de config (YAML + dict)."""

from pathlib import Path
from textwrap import dedent

import pytest

from src.config import (
    DataConfig,
    LoraConfig,
    QuantizationConfig,
    RunConfig,
    TrainingArgs,
    load_yaml_config,
    run_config_from_dict,
)


def test_default_run_config_is_valid():
    cfg = RunConfig()
    assert cfg.lora.r == 8
    assert cfg.lora.alpha == 16
    assert cfg.data.dataset == "alpaca_fr"
    assert cfg.uses_quantization is False


def test_run_config_from_dict_partial_overrides():
    cfg = run_config_from_dict(
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "lora": {"r": 32, "alpha": 64, "target_modules": ["q_proj", "k_proj"]},
            "quant": {"load_in_4bit": True},
        }
    )
    assert isinstance(cfg, RunConfig)
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 64
    assert cfg.lora.target_modules == ["q_proj", "k_proj"]
    assert cfg.uses_quantization is True
    # défauts conservés sur les sections non fournies
    assert cfg.data.dataset == "alpaca_fr"
    assert cfg.training.num_train_epochs == 1.0


def test_run_config_from_dict_filters_unknown_keys():
    cfg = run_config_from_dict({"lora": {"r": 4, "bogus_field": 999}})
    assert cfg.lora.r == 4


def test_run_config_from_dict_rejects_non_dict_payload():
    with pytest.raises(TypeError):
        run_config_from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_run_config_from_dict_rejects_non_dict_section():
    with pytest.raises(TypeError):
        run_config_from_dict({"lora": "r=8"})


def test_load_yaml_config_roundtrip(tmp_path: Path):
    yaml_text = dedent(
        """
        model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
        lora:
          r: 16
          alpha: 32
        quant:
          load_in_4bit: true
        training:
          output_dir: runs/test
          learning_rate: 1.0e-4
        """
    ).strip()
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)

    cfg = load_yaml_config(path)
    assert cfg.model_name.endswith("TinyLlama-1.1B-Chat-v1.0")
    assert cfg.lora.r == 16
    assert cfg.lora.alpha == 32
    assert cfg.uses_quantization is True
    assert cfg.training.output_dir == "runs/test"
    assert cfg.training.learning_rate == 1.0e-4


def test_to_dict_is_jsonable():
    cfg = RunConfig()
    d = cfg.to_dict()
    assert d["lora"]["r"] == 8
    assert d["training"]["output_dir"].startswith("runs/")


def test_bundled_yaml_configs_parse():
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(cfg_dir.glob("*.yaml")):
        cfg = load_yaml_config(path)
        assert isinstance(cfg, RunConfig)
