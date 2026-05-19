"""Export d'un modèle HF (merged) → GGUF via llama.cpp.

On ne ré-implémente pas la conversion : on délègue à `convert_hf_to_gguf.py`
fourni par le repo llama.cpp (https://github.com/ggerganov/llama.cpp). Cette
classe encapsule juste :
  - résolution du chemin du script (env var `LLAMA_CPP_DIR` ou flag CLI)
  - sélection du dtype de sortie (`f32`, `f16`, `bf16`, `auto`)
  - quantization optionnelle via le binaire `llama-quantize`

Usage :
    # 1) merge LoRA → modèle standalone (déjà fourni par src.merge)
    python -m src.merge --base-model ... --adapter ... --output merged/

    # 2) merged → GGUF
    python -m src.gguf_export \\
        --model merged/ \\
        --output models/qwen05b.f16.gguf \\
        --outtype f16 \\
        --llama-cpp-dir ~/llama.cpp

    # 3) optionnel : quantize en Q4_K_M
    python -m src.gguf_export \\
        --model merged/ \\
        --output models/qwen05b.q4_k_m.gguf \\
        --quantize Q4_K_M \\
        --llama-cpp-dir ~/llama.cpp
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

CONVERT_SCRIPT_NAME = "convert_hf_to_gguf.py"
QUANTIZE_BINARY_NAME = "llama-quantize"

VALID_OUTTYPES = ("f32", "f16", "bf16", "auto")
COMMON_QUANT_TYPES = (
    "Q4_K_M",
    "Q4_K_S",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
)


@dataclass(frozen=True)
class ExportRequest:
    model: Path
    output: Path
    outtype: str = "f16"
    llama_cpp_dir: Optional[Path] = None
    quantize: Optional[str] = None


def resolve_llama_cpp_dir(
    cli_value: Optional[Path],
    env: Optional[dict[str, str]] = None,
) -> Path:
    """Résout le chemin du repo llama.cpp.

    Ordre : flag CLI > env var LLAMA_CPP_DIR > ~/llama.cpp si présent.
    """
    if env is None:
        env = dict(os.environ)

    candidates: list[Optional[Path]] = [cli_value]
    if "LLAMA_CPP_DIR" in env:
        candidates.append(Path(env["LLAMA_CPP_DIR"]))
    candidates.append(Path.home() / "llama.cpp")

    for cand in candidates:
        if cand is None:
            continue
        if (cand / CONVERT_SCRIPT_NAME).is_file():
            return cand

    raise FileNotFoundError(
        "Could not find llama.cpp. Pass --llama-cpp-dir, set "
        "LLAMA_CPP_DIR, or clone https://github.com/ggerganov/llama.cpp "
        "into ~/llama.cpp."
    )


def build_convert_command(request: ExportRequest, llama_cpp_dir: Path) -> list[str]:
    """Construit la commande shell `python convert_hf_to_gguf.py ...`."""
    if request.outtype not in VALID_OUTTYPES:
        raise ValueError(
            f"outtype must be one of {VALID_OUTTYPES}, got {request.outtype!r}"
        )
    if not request.model.is_dir():
        raise FileNotFoundError(f"Model dir not found: {request.model}")

    script = llama_cpp_dir / CONVERT_SCRIPT_NAME
    return [
        "python",
        str(script),
        str(request.model),
        "--outfile",
        str(request.output),
        "--outtype",
        request.outtype,
    ]


def build_quantize_command(input_gguf: Path, output_gguf: Path, quant: str) -> list[str]:
    """Construit la commande pour `llama-quantize`."""
    binary = shutil.which(QUANTIZE_BINARY_NAME)
    if binary is None:
        raise FileNotFoundError(
            f"{QUANTIZE_BINARY_NAME} not found on PATH. Build llama.cpp first "
            "(`cmake --build build --target llama-quantize`) and ensure the "
            "binary is in PATH."
        )
    return [binary, str(input_gguf), str(output_gguf), quant]


def export_to_gguf(
    request: ExportRequest,
    *,
    runner: Callable[[list[str]], int] = lambda cmd: subprocess.run(cmd, check=True).returncode,
) -> Path:
    """Exécute la conversion HF→GGUF (puis quantize si demandé).

    `runner` est injectable pour les tests.
    """
    llama_cpp_dir = resolve_llama_cpp_dir(request.llama_cpp_dir)
    request.output.parent.mkdir(parents=True, exist_ok=True)

    convert_cmd = build_convert_command(request, llama_cpp_dir)
    runner(convert_cmd)

    if request.quantize:
        quantized_path = request.output.with_name(
            f"{request.output.stem}.{request.quantize.lower()}{request.output.suffix}"
        )
        quant_cmd = build_quantize_command(request.output, quantized_path, request.quantize)
        runner(quant_cmd)
        return quantized_path

    return request.output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-gguf-export",
        description="Export a merged HF model to GGUF via llama.cpp.",
    )
    parser.add_argument("--model", required=True, type=Path,
                        help="HF model dir (already merged, not an adapter).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .gguf path.")
    parser.add_argument("--outtype", default="f16", choices=VALID_OUTTYPES)
    parser.add_argument("--llama-cpp-dir", type=Path,
                        help="Path to llama.cpp repo (or use LLAMA_CPP_DIR env).")
    parser.add_argument("--quantize", choices=COMMON_QUANT_TYPES,
                        help="Optional post-conversion quantization.")
    return parser


def parse_args(argv: Sequence[str]) -> ExportRequest:
    args = _build_parser().parse_args(argv)
    return ExportRequest(
        model=args.model,
        output=args.output,
        outtype=args.outtype,
        llama_cpp_dir=args.llama_cpp_dir,
        quantize=args.quantize,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    request = parse_args(list(argv) if argv is not None else [])
    out = export_to_gguf(request)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main(sys.argv[1:]))
