"""Talea: sequence-only protein-solenoid discovery."""

from __future__ import annotations


__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TaleaPredictor":
        from talea.inference import TaleaPredictor

        return TaleaPredictor
    if name == "SequenceRecord":
        from talea.io import SequenceRecord

        return SequenceRecord
    raise AttributeError(name)


__all__ = ["SequenceRecord", "TaleaPredictor", "__version__"]
