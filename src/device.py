"""Détection device + dtypes adaptés à chaque backend.

Sépare la logique purement Python (testable) de l'utilisation de torch.
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol

DeviceKind = Literal["cuda", "mps", "cpu"]


class _TorchLike(Protocol):
    class cuda:  # noqa: N801
        @staticmethod
        def is_available() -> bool: ...

    class backends:  # noqa: N801
        class mps:  # noqa: N801
            @staticmethod
            def is_available() -> bool: ...


def detect_device(torch_module: Optional[_TorchLike] = None) -> DeviceKind:
    """Retourne le meilleur device dispo : cuda > mps > cpu.

    `torch_module` est injectable pour les tests.
    """
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_quantization(device: DeviceKind) -> bool:
    """bitsandbytes 4-bit / 8-bit nécessite CUDA."""
    return device == "cuda"


def resolve_precision_flags(
    device: DeviceKind,
    *,
    requested_bf16: bool,
    requested_fp16: bool,
) -> tuple[bool, bool]:
    """Force des flags compatibles avec le device.

    MPS ne supporte ni bf16 ni fp16 en training stable (au moins jusqu'à
    torch 2.8) → on retombe en float32. CPU idem, surtout sur Mac Intel.
    """
    if device in ("mps", "cpu"):
        return False, False
    return requested_bf16, requested_fp16
