"""Tests du rendu instruct → text pour SFTTrainer."""

from src.data.instruct import InstructSample
from src.formatting import render_batch, render_sample


SAMPLE = InstructSample(
    instruction="Traduis en anglais : Bonjour",
    response="Hello",
    source="alpaca_fr",
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


def test_render_sample_fallback_when_no_tokenizer():
    text = render_sample(SAMPLE)
    assert "### Instruction:" in text
    assert "### Response:" in text
    assert SAMPLE.instruction in text
    assert SAMPLE.response in text


def test_render_sample_uses_chat_template():
    text = render_sample(SAMPLE, _FakeTokenizer())
    assert text == f"user:{SAMPLE.instruction}|assistant:{SAMPLE.response}"


def test_render_sample_falls_back_when_template_raises():
    class Broken:
        def apply_chat_template(self, **_):
            raise RuntimeError("no template")

    text = render_sample(SAMPLE, Broken())
    assert "### Instruction:" in text


def test_render_batch_returns_text_dicts():
    out = render_batch([SAMPLE, SAMPLE])
    assert len(out) == 2
    assert all(set(d.keys()) == {"text"} for d in out)
