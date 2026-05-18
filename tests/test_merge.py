"""Tests merge LoRA — peft / transformers mockés, pas de download ni GPU."""

from pathlib import Path

import pytest

from src.merge import MergeRequest, merge_lora, parse_args


class _FakeMerged:
    def __init__(self):
        self.saved_to: str | None = None

    def save_pretrained(self, path: str, **_):
        self.saved_to = path


class _FakePeftModel:
    def __init__(self, base, adapter_path):
        self.base = base
        self.adapter_path = adapter_path
        self.merged = _FakeMerged()
        self.merge_called = False

    def merge_and_unload(self):
        self.merge_called = True
        return self.merged


class _FakeTokenizer:
    def __init__(self):
        self.saved_to: str | None = None

    def save_pretrained(self, path: str, **_):
        self.saved_to = path


def _make_fakes():
    state = {"peft": None, "tokenizer": _FakeTokenizer()}

    def base_loader(name, **kwargs):
        return {"base": name, "kwargs": kwargs}

    def peft_loader(base, adapter_path):
        state["peft"] = _FakePeftModel(base, adapter_path)
        return state["peft"]

    def tokenizer_loader(name):
        return state["tokenizer"]

    return state, base_loader, peft_loader, tokenizer_loader


def _make_request(tmp_path: Path, *, save_tokenizer: bool = True) -> MergeRequest:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    return MergeRequest(
        base_model="fake/base",
        adapter=adapter,
        output=tmp_path / "merged",
        dtype="bfloat16",
        save_tokenizer=save_tokenizer,
    )


def test_merge_lora_calls_merge_and_saves(tmp_path: Path):
    state, base_loader, peft_loader, tok_loader = _make_fakes()
    request = _make_request(tmp_path)

    out = merge_lora(
        request,
        base_loader=base_loader,
        peft_loader=peft_loader,
        tokenizer_loader=tok_loader,
    )

    assert out == request.output
    assert request.output.is_dir()
    assert state["peft"].merge_called is True
    assert state["peft"].merged.saved_to == str(request.output)
    assert state["tokenizer"].saved_to == str(request.output)


def test_merge_lora_skips_tokenizer_when_disabled(tmp_path: Path):
    state, base_loader, peft_loader, tok_loader = _make_fakes()
    request = _make_request(tmp_path, save_tokenizer=False)

    merge_lora(
        request,
        base_loader=base_loader,
        peft_loader=peft_loader,
        tokenizer_loader=tok_loader,
    )
    assert state["tokenizer"].saved_to is None


def test_merge_lora_skips_tokenizer_when_loader_missing(tmp_path: Path):
    state, base_loader, peft_loader, _ = _make_fakes()
    request = _make_request(tmp_path)

    merge_lora(
        request,
        base_loader=base_loader,
        peft_loader=peft_loader,
        tokenizer_loader=None,
    )
    assert state["tokenizer"].saved_to is None


def test_merge_lora_raises_when_adapter_missing(tmp_path: Path):
    request = MergeRequest(
        base_model="fake/base",
        adapter=tmp_path / "missing",
        output=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError):
        merge_lora(
            request,
            base_loader=lambda *a, **k: None,
            peft_loader=lambda *a, **k: None,
        )


def test_parse_args_basic():
    req = parse_args(
        [
            "--base-model",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--adapter",
            "runs/tinyllama_r8",
            "--output",
            "runs/tinyllama_r8_merged",
        ]
    )
    assert req.base_model.endswith("TinyLlama-1.1B-Chat-v1.0")
    assert req.adapter == Path("runs/tinyllama_r8")
    assert req.output == Path("runs/tinyllama_r8_merged")
    assert req.dtype == "bfloat16"
    assert req.save_tokenizer is True


def test_parse_args_no_tokenizer_flag():
    req = parse_args(
        [
            "--base-model",
            "x",
            "--adapter",
            "a",
            "--output",
            "o",
            "--no-tokenizer",
            "--dtype",
            "float16",
        ]
    )
    assert req.save_tokenizer is False
    assert req.dtype == "float16"
