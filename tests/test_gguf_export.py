"""Tests GGUF export — subprocess et filesystem mockés, pas de llama.cpp réel."""

from pathlib import Path

import pytest

from src.gguf_export import (
    CONVERT_SCRIPT_NAME,
    ExportRequest,
    build_convert_command,
    build_quantize_command,
    export_to_gguf,
    parse_args,
    resolve_llama_cpp_dir,
)


def _make_llama_cpp(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / CONVERT_SCRIPT_NAME).write_text("# stub")
    return root


# ---------- resolve_llama_cpp_dir ----------

def test_resolve_prefers_cli(tmp_path: Path):
    cli = _make_llama_cpp(tmp_path)
    env = {"LLAMA_CPP_DIR": "/nope"}
    assert resolve_llama_cpp_dir(cli, env=env) == cli


def test_resolve_falls_back_to_env(tmp_path: Path):
    root = _make_llama_cpp(tmp_path)
    env = {"LLAMA_CPP_DIR": str(root)}
    assert resolve_llama_cpp_dir(None, env=env) == root


def test_resolve_raises_when_nothing_found(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no_such_user")
    with pytest.raises(FileNotFoundError):
        resolve_llama_cpp_dir(None, env={})


# ---------- build_convert_command ----------

def test_build_convert_command_basic(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    llama_cpp = _make_llama_cpp(tmp_path)

    req = ExportRequest(model=model, output=tmp_path / "out.gguf", outtype="f16")
    cmd = build_convert_command(req, llama_cpp)

    assert cmd[0] == "python"
    assert cmd[1].endswith(CONVERT_SCRIPT_NAME)
    assert str(model) in cmd
    assert "--outfile" in cmd and str(tmp_path / "out.gguf") in cmd
    assert "--outtype" in cmd and "f16" in cmd


def test_build_convert_command_rejects_bad_outtype(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    llama_cpp = _make_llama_cpp(tmp_path)
    req = ExportRequest(model=model, output=tmp_path / "out.gguf", outtype="q4_0")
    with pytest.raises(ValueError):
        build_convert_command(req, llama_cpp)


def test_build_convert_command_requires_model_dir(tmp_path: Path):
    llama_cpp = _make_llama_cpp(tmp_path)
    req = ExportRequest(model=tmp_path / "missing", output=tmp_path / "out.gguf")
    with pytest.raises(FileNotFoundError):
        build_convert_command(req, llama_cpp)


# ---------- build_quantize_command ----------

def test_build_quantize_command_uses_binary(monkeypatch, tmp_path: Path):
    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr("shutil.which", lambda name: str(fake_bin))

    cmd = build_quantize_command(tmp_path / "in.gguf", tmp_path / "out.gguf", "Q4_K_M")
    assert cmd[0] == str(fake_bin)
    assert cmd[1].endswith("in.gguf")
    assert cmd[2].endswith("out.gguf")
    assert cmd[3] == "Q4_K_M"


def test_build_quantize_command_errors_when_binary_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        build_quantize_command(tmp_path / "i.gguf", tmp_path / "o.gguf", "Q4_K_M")


# ---------- end-to-end (mocked) ----------

def test_export_to_gguf_no_quant(monkeypatch, tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    llama_cpp = _make_llama_cpp(tmp_path)
    output = tmp_path / "out" / "model.f16.gguf"

    calls: list[list[str]] = []

    def fake_runner(cmd):
        calls.append(cmd)
        return 0

    req = ExportRequest(
        model=model, output=output, outtype="f16", llama_cpp_dir=llama_cpp,
    )
    out = export_to_gguf(req, runner=fake_runner)

    assert out == output
    assert output.parent.is_dir()
    assert len(calls) == 1
    assert calls[0][1].endswith(CONVERT_SCRIPT_NAME)


def test_export_to_gguf_with_quant(monkeypatch, tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    llama_cpp = _make_llama_cpp(tmp_path)
    output = tmp_path / "out" / "model.f16.gguf"

    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr("shutil.which", lambda name: str(fake_bin))

    calls: list[list[str]] = []

    def fake_runner(cmd):
        calls.append(cmd)
        return 0

    req = ExportRequest(
        model=model, output=output, outtype="f16",
        llama_cpp_dir=llama_cpp, quantize="Q4_K_M",
    )
    out = export_to_gguf(req, runner=fake_runner)

    assert out != output  # quantized file path
    assert "q4_k_m" in out.name
    assert len(calls) == 2
    assert calls[0][1].endswith(CONVERT_SCRIPT_NAME)
    assert calls[1][0] == str(fake_bin)
    assert calls[1][-1] == "Q4_K_M"


# ---------- argparse ----------

def test_parse_args_minimal():
    req = parse_args(["--model", "m/", "--output", "o.gguf"])
    assert req.model == Path("m/")
    assert req.output == Path("o.gguf")
    assert req.outtype == "f16"
    assert req.quantize is None


def test_parse_args_with_quant_and_outtype():
    req = parse_args([
        "--model", "m/", "--output", "o.gguf",
        "--outtype", "bf16", "--quantize", "Q5_K_M",
    ])
    assert req.outtype == "bf16"
    assert req.quantize == "Q5_K_M"
