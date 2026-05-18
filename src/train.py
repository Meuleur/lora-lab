"""Script paramétrable de fine-tuning LoRA / QLoRA.

CPU-friendly à l'import : tous les modules lourds (torch, transformers, peft, trl)
sont chargés *à l'intérieur* de `run_training` pour que la config + le CLI
restent testables sans GPU ni dépendances ML.

Usage :
    python -m src.train --config configs/tinyllama_r8.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import RunConfig, load_yaml_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-train",
        description="Fine-tune a small LM with LoRA / QLoRA via trl.SFTTrainer.",
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
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def run_training(config: RunConfig) -> None:  # pragma: no cover - needs GPU
    """Lance le fine-tuning. Non couvert par les tests (GPU requis)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    from src.data.instruct import SupportedDataset, load_instruct
    from src.formatting import render_batch

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.bfloat16 if config.training.bf16 else torch.float16}
    if config.uses_quantization:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.quant.load_in_4bit,
            load_in_8bit=config.quant.load_in_8bit,
            bnb_4bit_compute_dtype=getattr(torch, config.quant.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.quant.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quant.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    peft_cfg = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )

    samples = load_instruct(
        SupportedDataset(config.data.dataset),
        split=config.data.split,
        subset=config.data.subset,
    )
    dataset = Dataset.from_list(render_batch(samples, tokenizer))

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        seed=config.training.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_cfg,
        max_seq_length=config.data.max_seq_length,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model(config.training.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else [])
    config = load_yaml_config(args.config)
    if args.dry_run:
        print(config.to_dict())
        return 0
    run_training(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main(sys.argv[1:]))
