"""Sampling depuis un base model + LoRA adapter, pour évaluation qualitative.

Usage :
    python -m src.generate \
        --base-model Qwen/Qwen2.5-0.5B \
        --adapter runs/qwen05b_lora_mps \
        --output runs/qwen05b_lora_mps/sample_generations.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence


DEFAULT_PROMPTS: List[str] = [
    "Explique l'effet de serre en quelques phrases simples.",
    "Donne-moi une recette rapide de pâtes à la tomate.",
    "Quelle est la capitale de l'Australie ?",
    "Écris un court poème sur l'automne.",
    "Liste trois conseils pour mieux dormir.",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lora-generate")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser


def _format_prompt(instruction: str) -> str:
    """Même template que celui utilisé à l'entraînement (fallback alpaca-style)."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _strip_response(full: str, prompt: str) -> str:
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    # certains tokenizers réinjectent un BOS — on coupe après "### Response:"
    marker = "### Response:"
    idx = full.rfind(marker)
    if idx != -1:
        return full[idx + len(marker):].strip()
    return full.strip()


def generate_samples(  # pragma: no cover - real models/MPS
    base_model: str,
    adapter: Path,
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.device import detect_device

    device = detect_device(torch)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device).eval()

    results = []
    for prompt in prompts:
        text = _format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - t0
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        response = _strip_response(decoded, text)
        results.append({
            "prompt": prompt,
            "response": response,
            "seconds": elapsed,
            "new_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
        })
    return results


def render_markdown(base_model: str, adapter: Path, results, device: str) -> str:  # pragma: no cover - thin
    lines = [
        f"# Sample generations — {adapter.name}",
        "",
        f"- **base model**: `{base_model}`",
        f"- **adapter**: `{adapter}`",
        f"- **device**: `{device}`",
        f"- **prompts**: {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.extend([
            f"## Prompt {i}",
            "",
            f"> {r['prompt']}",
            "",
            "**Réponse :**",
            "",
            "```",
            r["response"],
            "```",
            "",
            f"_{r['new_tokens']} new tokens in {r['seconds']:.2f}s "
            f"({r['new_tokens'] / r['seconds']:.2f} tok/s)_",
            "",
        ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - real models
    import torch
    from src.device import detect_device

    args = _build_parser().parse_args(list(argv) if argv is not None else [])
    results = generate_samples(
        args.base_model,
        args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    device = detect_device(torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(args.base_model, args.adapter, results, device))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main(sys.argv[1:]))
