"""Canonical amino-acid sequence contracts used throughout Talea.

Talea does not impute, mask, or tokenize ambiguous residues. Every scored
sequence must contain only the 20 standard amino-acid one-letter codes.
"""

from __future__ import annotations

from collections.abc import Mapping


CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_AMINO_ACID_SET = frozenset(CANONICAL_AMINO_ACIDS)


def noncanonical_residues(sequence: str) -> tuple[str, ...]:
    """Return every distinct symbol outside the canonical amino-acid alphabet."""

    normalized = str(sequence).upper()
    return tuple(sorted(set(normalized) - CANONICAL_AMINO_ACID_SET))


def require_canonical_sequence(
    sequence: str,
    *,
    record_id: str | None = None,
    allow_empty: bool = False,
) -> str:
    """Uppercase and validate one protein sequence without residue imputation."""

    normalized = str(sequence).upper()
    label = f" for {record_id}" if record_id is not None else ""
    if not normalized and not allow_empty:
        raise ValueError(f"empty amino-acid sequence{label}")
    unexpected = noncanonical_residues(normalized)
    if unexpected:
        raise ValueError(
            f"noncanonical amino-acid residues{label}: {''.join(unexpected)}"
        )
    return normalized


def require_canonical_sequences(sequences: Mapping[str, str]) -> dict[str, str]:
    """Validate a mapping of record IDs to sequences and return uppercase copies."""

    return {
        str(record_id): require_canonical_sequence(
            sequence, record_id=str(record_id)
        )
        for record_id, sequence in sequences.items()
    }


def normalize_fasta_protein_sequence(
    sequence: str,
    *,
    record_id: str | None = None,
) -> str:
    """Normalize one optional terminal FASTA stop marker, then validate residues.

    A single final ``*`` is treated as a serialization marker for a known stop
    codon, not as an amino-acid residue. No internal or repeated stop marker is
    accepted, and no amino-acid ambiguity code is altered or imputed.
    """

    normalized = str(sequence).upper()
    if normalized.endswith("*"):
        normalized = normalized[:-1]
    return require_canonical_sequence(normalized, record_id=record_id)
