"""Script paramétrable de fine-tuning LoRA (CUDA / MPS / CPU).

CPU-friendly à l'import : tous les modules lourds (torch, transformers, peft, trl)
sont chargés *à l'intérieur* de `run_training`, donc la config + le CLI restent
testables sans dépendances ML.

QLoRA via bitsandbytes ne fonctionne qu'avec CUDA : sur MPS / CPU on warn et on
tombe sur du LoRA float32 (MPS) ou float32 (CPU). bf16 / fp16 sont désactivés
hors CUDA, ils ne sont pas stables.

Usage :
    python -m src.train --config configs/qwen05b_lora_mps.yaml --max-steps 100
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence

from src.config import RunConfig, load_yaml_config
from src.device import detect_device, resolve_precision_flags, supports_quantization

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-train",
        description="Fine-tune a small LM with LoRA via trl.SFTTrainer (CUDA/MPS/CPU).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML run config (see configs/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the config and print it without running training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap training to N optimizer steps (overrides num_train_epochs).",
    )
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _setup_file_logger(output_dir: Path) -> logging.Handler:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def run_training(  # pragma: no cover - needs real model + GPU/MPS
    config: RunConfig,
    *,
    max_steps: Optional[int] = None,
) -> Path:
    """Lance le fine-tuning et renvoie le dossier de sortie."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from src.data.instruct import SupportedDataset, load_instruct
    from src.formatting import render_batch

    output_dir = Path(config.training.output_dir)
    handler = _setup_file_logger(output_dir)

    device = detect_device(torch)
    bf16, fp16 = resolve_precision_flags(
        device,
        requested_bf16=config.training.bf16,
        requested_fp16=config.training.fp16,
    )
    logger.info("device=%s bf16=%s fp16=%s", device, bf16, fp16)

    if config.uses_quantization and not supports_quantization(device):
        warnings.warn(
            f"Quantization requested but device={device}: bitsandbytes needs CUDA. "
            "Falling back to plain LoRA (no 4-/8-bit).",
            stacklevel=2,
        )
        logger.warning("Skipping bitsandbytes — not supported on %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {}
    if config.uses_quantization and supports_quantization(device):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.quant.load_in_4bit,
            load_in_8bit=config.quant.load_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, config.quant.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.quant.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quant.bnb_4bit_use_double_quant,
        )
        model_kwargs["device_map"] = "auto"
    else:
        # MPS et CPU : float32 par défaut. On déplacera ensuite manuellement.
        model_kwargs["dtype"] = torch.float32

    logger.info("loading base model: %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if device == "mps":
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")

    peft_cfg = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

    logger.info("loading dataset: %s split=%s subset=%s",
                config.data.dataset, config.data.split, config.data.subset)
    samples = load_instruct(
        SupportedDataset(config.data.dataset),
        split=config.data.split,
        subset=config.data.subset,
    )
    dataset = Dataset.from_list(render_batch(samples, tokenizer))
    logger.info("dataset size: %d examples", len(dataset))

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=bf16,
        fp16=fp16,
        seed=config.training.seed,
        report_to=[],
        gradient_checkpointing=False,
        max_length=config.data.max_seq_length,
        dataset_text_field="text",
        max_steps=max_steps if max_steps is not None else -1,
        save_strategy="steps",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

    logger.info("starting training (max_steps=%s)", max_steps)
    train_result = trainer.train()
    logger.info("train metrics: %s", train_result.metrics)

    trainer.save_model(str(output_dir))
    logger.info("saved adapter to %s", output_dir)

    logging.getLogger().removeHandler(handler)
    handler.close()
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else [])
    config = load_yaml_config(args.config)
    if args.dry_run:
        print(config.to_dict())
        return 0
    run_training(config, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main(sys.argv[1:]))
