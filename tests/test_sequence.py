import pytest

from talea.sequence import (
    normalize_fasta_protein_sequence,
    noncanonical_residues,
    require_canonical_sequence,
    require_canonical_sequences,
)


@pytest.mark.parametrize("residue", ["X", "B", "J", "O", "U", "Z", "*", "-"])
def test_every_noncanonical_symbol_is_rejected(residue):
    with pytest.raises(ValueError, match="noncanonical"):
        require_canonical_sequence(f"ACD{residue}EFG", record_id="test")


def test_canonical_lowercase_is_normalized_without_imputation():
    assert require_canonical_sequence("acdefghiklmnpqrstvwy") == "ACDEFGHIKLMNPQRSTVWY"


def test_sequence_mapping_reports_record_id():
    with pytest.raises(ValueError, match="bad_record"):
        require_canonical_sequences({"good": "ACD", "bad_record": "ACX"})


def test_noncanonical_residues_are_unique_and_sorted():
    assert noncanonical_residues("AXZXXB") == ("B", "X", "Z")


def test_terminal_fasta_stop_marker_is_normalized_before_residue_validation():
    assert normalize_fasta_protein_sequence("ACDEF*") == "ACDEF"


@pytest.mark.parametrize("sequence", ["AC*DEF", "ACDEF**", "ACXDEF*"])
def test_fasta_normalization_never_hides_noncanonical_residues(sequence):
    with pytest.raises(ValueError, match="noncanonical"):
        normalize_fasta_protein_sequence(sequence)
