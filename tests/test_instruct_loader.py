"""Tests du loader unifié — backends mockés, pas de download HF."""

import pytest

from src.data.instruct import (
    InstructSample,
    SupportedDataset,
    load_alpaca_fr,
    load_instruct,
    load_oasst_fr,
    to_chat_messages,
)


# -----------------------------
# Alpaca-FR
# -----------------------------

ALPACA_ROWS = [
    {
        "instruction": "Écris un poème sur la mer",
        "input": "",
        "output": "Les vagues dansent au crépuscule…",
    },
    {
        "instruction": "Traduis en anglais",
        "input": "Bonjour le monde",
        "output": "Hello, world",
    },
    {
        "instruction": "Vide réponse",
        "input": "",
        "output": "",  # à filtrer
    },
    {
        "instruction": "",
        "input": "rien",
        "output": "réponse sans question",  # à filtrer
    },
]


def _alpaca_loader(name, split):
    assert name == "bofenghuang/vigogne-instruct-66k"
    assert split == "train"
    return list(ALPACA_ROWS)


def test_load_alpaca_fr_skips_empty_rows():
    samples = load_alpaca_fr(loader=_alpaca_loader)
    assert len(samples) == 2
    assert all(isinstance(s, InstructSample) for s in samples)
    assert all(s.source == "alpaca_fr" for s in samples)


def test_load_alpaca_fr_merges_input_into_instruction():
    samples = load_alpaca_fr(loader=_alpaca_loader)
    second = next(s for s in samples if "Traduis" in s.instruction)
    assert "Bonjour le monde" in second.instruction


def test_load_alpaca_fr_subset_caps_output():
    samples = load_alpaca_fr(subset=1, loader=_alpaca_loader)
    assert len(samples) == 1


# -----------------------------
# OASST-FR
# -----------------------------

OASST_ROWS = [
    {"message_id": "1", "parent_id": None, "role": "prompter", "text": "Salut", "lang": "fr"},
    {"message_id": "2", "parent_id": "1", "role": "assistant", "text": "Bonjour!", "lang": "fr"},
    {"message_id": "3", "parent_id": None, "role": "prompter", "text": "Hi", "lang": "en"},
    {"message_id": "4", "parent_id": "3", "role": "assistant", "text": "Hello", "lang": "en"},
    {"message_id": "5", "parent_id": "1", "role": "assistant", "text": "Hola", "lang": "es"},
    {"message_id": "6", "parent_id": None, "role": "prompter", "text": "Comment ça va ?", "lang": "fr"},
    {"message_id": "7", "parent_id": "6", "role": "assistant", "text": "Ça va bien", "lang": "fr"},
    # orphelin (parent inexistant)
    {"message_id": "8", "parent_id": "999", "role": "assistant", "text": "perdu", "lang": "fr"},
]


def _oasst_loader(name, split):
    assert name == "OpenAssistant/oasst1"
    return list(OASST_ROWS)


def test_load_oasst_fr_keeps_only_fr_pairs():
    samples = load_oasst_fr(loader=_oasst_loader)
    assert len(samples) == 2
    instrs = [s.instruction for s in samples]
    assert "Salut" in instrs
    assert "Comment ça va ?" in instrs


def test_load_oasst_fr_drops_orphans_and_other_langs():
    samples = load_oasst_fr(loader=_oasst_loader)
    for s in samples:
        assert "Hello" not in s.response
        assert "Hola" not in s.response
        assert "perdu" not in s.response


def test_load_oasst_fr_marks_source():
    samples = load_oasst_fr(loader=_oasst_loader)
    assert all(s.source == "oasst_fr" for s in samples)


# -----------------------------
# Dispatcher + formatter
# -----------------------------

def test_load_instruct_dispatcher_alpaca():
    samples = load_instruct(SupportedDataset.ALPACA_FR, loader=_alpaca_loader)
    assert all(s.source == "alpaca_fr" for s in samples)


def test_load_instruct_dispatcher_oasst():
    samples = load_instruct(SupportedDataset.OASST_FR, loader=_oasst_loader)
    assert all(s.source == "oasst_fr" for s in samples)


def test_load_instruct_raises_on_unknown():
    with pytest.raises(ValueError):
        load_instruct("not_a_dataset")  # type: ignore[arg-type]


def test_to_chat_messages_shape():
    s = InstructSample(instruction="Q?", response="A!", source="alpaca_fr")
    msgs = to_chat_messages(s)
    assert msgs == [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A!"},
    ]
