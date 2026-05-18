"""Tests utilitaires generate — pas de modèle réel ici."""

from src.generate import _format_prompt, _strip_response


def test_format_prompt_uses_alpaca_template():
    text = _format_prompt("Salut")
    assert "### Instruction:\nSalut" in text
    assert text.endswith("### Response:\n")


def test_strip_response_when_starts_with_prompt():
    prompt = "### Instruction:\nQ?\n\n### Response:\n"
    full = prompt + "Voici la réponse."
    assert _strip_response(full, prompt) == "Voici la réponse."


def test_strip_response_falls_back_on_marker():
    prompt = "ignored"
    full = "extra\n### Response:\nAlors la réponse"
    assert _strip_response(full, prompt) == "Alors la réponse"
