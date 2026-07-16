"""Strict FASTA input and JSONL output helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from talea.sequence import normalize_fasta_protein_sequence


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    sequence: str
    sequence_sha256: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    def metadata(self) -> dict[str, str | int]:
        return {
            "record_id": self.record_id,
            "length": self.length,
            "sequence_sha256": self.sequence_sha256,
        }


def _record(record_id: str, parts: list[str], maximum_length: int) -> SequenceRecord:
    sequence = normalize_fasta_protein_sequence(
        "".join(parts), record_id=record_id
    )
    if len(sequence) > maximum_length:
        raise ValueError(
            f"sequence {record_id} has {len(sequence)} residues; "
            f"Talea's frozen maximum is {maximum_length}"
        )
    return SequenceRecord(
        record_id=record_id,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
    )


def read_fasta(path: str | Path, maximum_length: int = 2046) -> list[SequenceRecord]:
    """Read a protein FASTA and reject every noncanonical record in full."""

    fasta = Path(path)
    if maximum_length < 1:
        raise ValueError("maximum length must be positive")
    records: list[SequenceRecord] = []
    identifiers: set[str] = set()
    record_id: str | None = None
    parts: list[str] = []
    with fasta.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    records.append(_record(record_id, parts, maximum_length))
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at line {line_number}")
                record_id = header.split()[0]
                if record_id in identifiers:
                    raise ValueError(f"duplicate FASTA record ID: {record_id}")
                identifiers.add(record_id)
                parts = []
            else:
                if record_id is None:
                    raise ValueError(
                        f"sequence data precedes the first header at line {line_number}"
                    )
                parts.append(line)
    if record_id is not None:
        records.append(_record(record_id, parts, maximum_length))
    if not records:
        raise ValueError(f"no FASTA records found in {fasta}")
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    output = Path(path)
    count = 0
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count
