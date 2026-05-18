"""Tests détection device + précision."""

import pytest

from src.device import detect_device, resolve_precision_flags, supports_quantization


class _FakeTorch:
    def __init__(self, cuda: bool, mps: bool):
        outer = self

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return outer._cuda

        class _Mps:
            @staticmethod
            def is_available() -> bool:
                return outer._mps

        class _Backends:
            mps = _Mps()

        self._cuda = cuda
        self._mps = mps
        self.cuda = _Cuda()
        self.backends = _Backends()


def test_detect_returns_one_of_known_kinds():
    # call without injection: must return one of cuda/mps/cpu on a real torch
    pytest.importorskip("torch")
    kind = detect_device()
    assert kind in {"cuda", "mps", "cpu"}


def test_detect_prefers_cuda():
    assert detect_device(_FakeTorch(cuda=True, mps=True)) == "cuda"
    assert detect_device(_FakeTorch(cuda=True, mps=False)) == "cuda"


def test_detect_mps_when_no_cuda():
    assert detect_device(_FakeTorch(cuda=False, mps=True)) == "mps"


def test_detect_cpu_fallback():
    assert detect_device(_FakeTorch(cuda=False, mps=False)) == "cpu"


def test_quantization_only_on_cuda():
    assert supports_quantization("cuda") is True
    assert supports_quantization("mps") is False
    assert supports_quantization("cpu") is False


def test_precision_disabled_on_mps_and_cpu():
    for dev in ("mps", "cpu"):
        bf16, fp16 = resolve_precision_flags(dev, requested_bf16=True, requested_fp16=True)
        assert (bf16, fp16) == (False, False)


def test_precision_passthrough_on_cuda():
    assert resolve_precision_flags("cuda", requested_bf16=True, requested_fp16=False) == (True, False)
    assert resolve_precision_flags("cuda", requested_bf16=False, requested_fp16=True) == (False, True)
