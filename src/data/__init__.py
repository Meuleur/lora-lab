"""Unified instruct dataset loaders."""

from src.data.instruct import (
    InstructSample,
    SupportedDataset,
    load_alpaca_fr,
    load_instruct,
    load_oasst_fr,
    to_chat_messages,
)

__all__ = [
    "InstructSample",
    "SupportedDataset",
    "load_alpaca_fr",
    "load_instruct",
    "load_oasst_fr",
    "to_chat_messages",
]
