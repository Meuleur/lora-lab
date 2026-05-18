"""Formattage des samples instruct → texte pour SFTTrainer.

Format Llama/Qwen-friendly : on délègue au chat_template du tokenizer si dispo,
sinon on tombe sur un template texte minimaliste.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol

from src.data.instruct import InstructSample, to_chat_messages


class _TokenizerWithTemplate(Protocol):
    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


_FALLBACK_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


def render_sample(
    sample: InstructSample,
    tokenizer: Optional[_TokenizerWithTemplate] = None,
) -> str:
    """Renvoie le texte d'entraînement pour un sample."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                to_chat_messages(sample),
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return _FALLBACK_TEMPLATE.format(
        instruction=sample.instruction,
        response=sample.response,
    )


def render_batch(
    samples: Iterable[InstructSample],
    tokenizer: Optional[_TokenizerWithTemplate] = None,
) -> List[Dict[str, str]]:
    """Renvoie une liste de dicts `{"text": ...}` consommables par SFTTrainer."""
    return [{"text": render_sample(s, tokenizer)} for s in samples]
