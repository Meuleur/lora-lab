"""Merge d'un adapter LoRA dans son base model et sauvegarde standalone.

Couche fine au-dessus de `peft.PeftModel.from_pretrained(...).merge_and_unload()`,
exposée en CLI :

    python -m src.merge \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter runs/tinyllama_r8 \
        --output runs/tinyllama_r8_merged

Les loaders (transformers + peft) sont injectables pour permettre des tests
sans GPU ni download.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence


class _Saveable(Protocol):
    def save_pretrained(self, path: str, **kwargs: Any) -> None: ...


@dataclass(frozen=True)
class MergeRequest:
    base_model: str
    adapter: Path
    output: Path
    dtype: str = "bfloat16"
    save_tokenizer: bool = True


def merge_lora(
    request: MergeRequest,
    *,
    base_loader: Callable[..., Any],
    peft_loader: Callable[..., Any],
    tokenizer_loader: Optional[Callable[..., _Saveable]] = None,
) -> Path:
    """Charge la base, applique l'adapter, merge, sauvegarde.

    Args:
        request: paramètres du merge.
        base_loader: callable type `AutoModelForCausalLM.from_pretrained`.
        peft_loader: callable type `PeftModel.from_pretrained(base, adapter_path)`.
        tokenizer_loader: callable type `AutoTokenizer.from_pretrained`.
                          Si None, on saute la sauvegarde tokenizer.

    Returns:
        Path du dossier de sortie.
    """
    if not request.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {request.adapter}")

    request.output.mkdir(parents=True, exist_ok=True)

    base = base_loader(request.base_model, torch_dtype=request.dtype)
    peft_model = peft_loader(base, str(request.adapter))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(request.output))

    if request.save_tokenizer and tokenizer_loader is not None:
        tok = tokenizer_loader(request.base_model)
        tok.save_pretrained(str(request.output))

    return request.output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-merge",
        description="Merge a trained LoRA adapter into its base model.",
    )
    parser.add_argument("--base-model", required=True, help="HF model id of the base model.")
    parser.add_argument("--adapter", required=True, type=Path, help="Path to the LoRA adapter dir.")
    parser.add_argument("--output", required=True, type=Path, help="Where to save the merged model.")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--no-tokenizer", action="store_true", help="Skip tokenizer save.")
    return parser


def parse_args(argv: Sequence[str]) -> MergeRequest:
    args = _build_parser().parse_args(argv)
    return MergeRequest(
        base_model=args.base_model,
        adapter=args.adapter,
        output=args.output,
        dtype=args.dtype,
        save_tokenizer=not args.no_tokenizer,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs deps
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    request = parse_args(list(argv) if argv is not None else [])
    merge_lora(
        request,
        base_loader=AutoModelForCausalLM.from_pretrained,
        peft_loader=PeftModel.from_pretrained,
        tokenizer_loader=AutoTokenizer.from_pretrained,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main(sys.argv[1:]))
