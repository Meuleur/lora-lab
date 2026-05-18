"""Dataclasses de config + chargement YAML pour les runs LoRA.

Conçu pour rester testable sans torch / transformers : pas d'imports lourds ici.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass(frozen=True)
class TrainingArgs:
    output_dir: str = "runs/default"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    bf16: bool = True
    fp16: bool = False
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "alpaca_fr"  # voir SupportedDataset
    split: str = "train"
    subset: Optional[int] = None
    max_seq_length: int = 1024


@dataclass(frozen=True)
class RunConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    quant: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingArgs = field(default_factory=TrainingArgs)

    @property
    def uses_quantization(self) -> bool:
        return self.quant.load_in_4bit or self.quant.load_in_8bit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_NESTED_TYPES: Dict[str, type] = {
    "data": DataConfig,
    "lora": LoraConfig,
    "quant": QuantizationConfig,
    "training": TrainingArgs,
}


def _build_nested(payload: Dict[str, Any], cls: type) -> Any:
    """Filtre les clés inconnues plutôt que de planter — pratique pour évoluer."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in allowed})


def run_config_from_dict(payload: Dict[str, Any]) -> RunConfig:
    """Construit une RunConfig à partir d'un dict (parsing YAML / JSON)."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    kwargs: Dict[str, Any] = {}
    if "model_name" in payload:
        kwargs["model_name"] = payload["model_name"]

    for key, cls in _NESTED_TYPES.items():
        if key in payload and payload[key] is not None:
            sub = payload[key]
            if not isinstance(sub, dict):
                raise TypeError(f"'{key}' must be a dict, got {type(sub).__name__}")
            kwargs[key] = _build_nested(sub, cls)

    return RunConfig(**kwargs)


def load_yaml_config(path: str | Path) -> RunConfig:
    """Charge un YAML disque → RunConfig."""
    import yaml  # import paresseux

    path = Path(path)
    with path.open() as f:
        payload = yaml.safe_load(f) or {}
    return run_config_from_dict(payload)
