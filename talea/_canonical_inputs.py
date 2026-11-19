"""Exact host input tables for the frozen ESM++ character tokenizer."""
import hashlib
import inspect
import json

import numpy as np
import torch

_ALPHABET = b"ACDEFGHIKLMNPQRSTVWY"
_BACKEND_SHA256 = "e38b47102e48b15efb28f6aac834b704c9a5d98e89bd031866439d684cef901b"
_CLASS_SHA256 = "430019385418e039fd3fee4634461d49ecd8159ce9c3155e47e40d049c321d78"


def canonical_token_table(tokenizer):
    """Use the direct table only for the verified tokenizer and its normal padding contract."""
    if (type(tokenizer).__name__ != "EsmSequenceTokenizer" or tokenizer.padding_side != "right"
            or list(tokenizer.model_input_names) != ["input_ids", "attention_mask"]):
        return None
    try:
        source = inspect.getsource(type(tokenizer)).encode()
    except (OSError, TypeError):
        return None
    if hashlib.sha256(source).hexdigest() != _CLASS_SHA256:
        return None
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # A normal padding=True, truncation=False call reconfigures these two fields.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    serialized = json.dumps(backend, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(serialized).hexdigest() != _BACKEND_SHA256:
        return None
    ids = bytes(tokenizer.convert_tokens_to_ids([chr(c) for c in _ALPHABET]))
    # Also check the effective Python wrapper, including its special-token overrides.
    example = tokenizer([_ALPHABET.decode(), "A"], padding=True, return_special_tokens_mask=True)
    width = len(_ALPHABET) + 2
    cls, eos, pad = tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    expected = {
        "input_ids": [[cls, *ids, eos], [cls, ids[0], eos] + [pad] * (width - 3)],
        "attention_mask": [[1] * width, [1] * 3 + [0] * (width - 3)],
        "special_tokens_mask": [[1] + [0] * len(ids) + [1], [1, 0] + [1] * (width - 2)],
    }
    if dict(example) != expected:
        return None
    return bytes.maketrans(_ALPHABET, ids), cls, eos, pad


def canonical_batch(records, table, device):
    """Encode validated canonical proteins. Every residue span starts just after CLS."""
    mapping, cls, eos, pad = table
    longest = max(r.length for r in records)
    ids = np.full((len(records), longest + 2), pad, dtype=np.int64)
    mask = np.zeros_like(ids)
    ids[:, 0] = cls
    positions = torch.arange(1, longest + 1, dtype=torch.long).to(device)
    spans = []
    for i, record in enumerate(records):
        end = record.length + 1
        ids[i, 1:end] = np.frombuffer(record.sequence.encode("ascii").translate(mapping), dtype=np.uint8)
        ids[i, end] = eos
        mask[i, :end + 1] = 1
        spans.append((positions[:record.length], slice(1, end)))
    return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}, spans
