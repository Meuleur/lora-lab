"""Loader unifié pour datasets d'instructions FR.

Sources supportées :
  - Alpaca-FR : `bofenghuang/vigogne-instruct-66k` ou équivalent
                (champs : instruction, input, output)
  - OpenAssistant-FR : `OpenAssistant/oasst1` filtré sur `lang == "fr"`
                       (structure conversationnelle parent/child)

Sortie commune : `InstructSample` (instruction libre + réponse) que l'on peut
ensuite formatter en chat messages pour `trl.SFTTrainer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional

ALPACA_FR_NAME = "jpacifico/French-Alpaca-dataset-Instruct-55K"
OASST_NAME = "OpenAssistant/oasst1"


class SupportedDataset(str, Enum):
    ALPACA_FR = "alpaca_fr"
    OASST_FR = "oasst_fr"


@dataclass(frozen=True)
class InstructSample:
    instruction: str
    response: str
    source: str  # "alpaca_fr" | "oasst_fr"


def _coerce(text: object) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _alpaca_row_to_sample(row: Dict) -> Optional[InstructSample]:
    instr = _coerce(row.get("instruction"))
    extra = _coerce(row.get("input"))
    response = _coerce(row.get("output"))
    if not instr or not response:
        return None
    full_instr = f"{instr}\n\n{extra}".strip() if extra else instr
    return InstructSample(
        instruction=full_instr,
        response=response,
        source=SupportedDataset.ALPACA_FR.value,
    )


def load_alpaca_fr(
    split: str = "train",
    subset: Optional[int] = None,
    loader: Optional[Callable[..., Iterable[Dict]]] = None,
) -> List[InstructSample]:
    """Charge Alpaca-FR (Vigogne) sous forme `InstructSample`."""
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    raw = loader(ALPACA_FR_NAME, split=split)
    out: List[InstructSample] = []
    for i, row in enumerate(raw):
        if subset is not None and len(out) >= subset:
            break
        sample = _alpaca_row_to_sample(dict(row))
        if sample is not None:
            out.append(sample)
        _ = i
    return out


def _oasst_pairs_from_messages(rows: List[Dict]) -> List[InstructSample]:
    """Reconstitue les paires (prompt → réponse assistant) depuis OASST.

    Chaque message a `message_id`, `parent_id`, `role` (`prompter` | `assistant`),
    `text` et `lang`. On garde les paires prompter→assistant en FR seulement.
    """
    by_id: Dict[str, Dict] = {r["message_id"]: r for r in rows if r.get("message_id")}
    out: List[InstructSample] = []
    for row in rows:
        if row.get("role") != "assistant":
            continue
        if row.get("lang") and row["lang"] != "fr":
            continue
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent.get("role") != "prompter":
            continue
        if parent.get("lang") and parent["lang"] != "fr":
            continue
        instr = _coerce(parent.get("text"))
        resp = _coerce(row.get("text"))
        if not instr or not resp:
            continue
        out.append(
            InstructSample(
                instruction=instr,
                response=resp,
                source=SupportedDataset.OASST_FR.value,
            )
        )
    return out


def load_oasst_fr(
    split: str = "train",
    subset: Optional[int] = None,
    loader: Optional[Callable[..., Iterable[Dict]]] = None,
) -> List[InstructSample]:
    """Charge OpenAssistant-FR sous forme `InstructSample`."""
    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    raw = list(loader(OASST_NAME, split=split))
    pairs = _oasst_pairs_from_messages([dict(r) for r in raw])
    if subset is not None:
        pairs = pairs[:subset]
    return pairs


def load_instruct(
    dataset: SupportedDataset,
    split: str = "train",
    subset: Optional[int] = None,
    loader: Optional[Callable[..., Iterable[Dict]]] = None,
) -> List[InstructSample]:
    """Dispatcher unifié."""
    if dataset == SupportedDataset.ALPACA_FR:
        return load_alpaca_fr(split=split, subset=subset, loader=loader)
    if dataset == SupportedDataset.OASST_FR:
        return load_oasst_fr(split=split, subset=subset, loader=loader)
    raise ValueError(f"Unsupported dataset: {dataset}")


def to_chat_messages(sample: InstructSample) -> List[Dict[str, str]]:
    """Formate un sample en messages chat (compatible templates HF)."""
    return [
        {"role": "user", "content": sample.instruction},
        {"role": "assistant", "content": sample.response},
    ]
