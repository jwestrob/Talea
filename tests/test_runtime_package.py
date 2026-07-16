import hashlib

import pytest

from talea.artifacts import (
    DEPLOYMENT_SHA256,
    load_deployment,
    sha256_file,
    verified_runtime_assets,
)
from talea.cli import main
from talea.io import read_fasta


def test_deployment_and_runtime_artifacts_verify() -> None:
    deployment = load_deployment()
    assets = verified_runtime_assets(deployment)
    assert deployment["schema_version"] == "talea_stable_discovery_v2"
    assert sha256_file(assets.deployment_path) == DEPLOYMENT_SHA256
    assert set(assets.sha256) == {
        "teacher_model",
        "teacher_contract",
        "global_model",
        "protein_calibration",
        "local_calibration",
        "wedge_config",
        "topology_config",
    }


def test_fasta_reader_normalizes_one_terminal_stop(tmp_path) -> None:
    fasta = tmp_path / "proteins.faa"
    fasta.write_text(">a description\nACDEFG*\n>b\nHIKLMN\n")
    records = read_fasta(fasta)
    assert [record.record_id for record in records] == ["a", "b"]
    assert records[0].sequence == "ACDEFG"
    assert records[0].sequence_sha256 == hashlib.sha256(b"ACDEFG").hexdigest()


@pytest.mark.parametrize("sequence", ["ACDX", "ACD**", "ACD*EF", ""])
def test_fasta_reader_rejects_invalid_sequences(tmp_path, sequence: str) -> None:
    fasta = tmp_path / "invalid.faa"
    fasta.write_text(f">bad\n{sequence}\n")
    with pytest.raises(ValueError):
        read_fasta(fasta)


def test_fasta_reader_rejects_duplicate_ids(tmp_path) -> None:
    fasta = tmp_path / "duplicate.faa"
    fasta.write_text(">same first\nACDE\n>same second\nFGHI\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_fasta(fasta)


def test_fasta_reader_enforces_frozen_maximum(tmp_path) -> None:
    deployment = load_deployment()
    maximum = int(deployment["input_contract"]["maximum_residues"])
    fasta = tmp_path / "long.faa"
    fasta.write_text(">long\n" + "A" * (maximum + 1) + "\n")
    with pytest.raises(ValueError, match="maximum"):
        read_fasta(fasta, maximum_length=maximum)


def test_runtime_contract_has_no_hard_threshold() -> None:
    deployment = load_deployment()
    assert deployment["reporting_contract"]["rank_by"] == "talea_discovery_score"
    assert deployment["reporting_contract"]["hard_solenoid_threshold"] is None
    assert deployment["input_contract"]["maximum_noncanonical_residues"] == 0


def test_cli_refuses_to_overwrite_its_input(tmp_path) -> None:
    fasta = tmp_path / "protein.faa"
    fasta.write_text(">a\nACDEFG\n")
    with pytest.raises(ValueError, match="different files"):
        main(["--input", str(fasta), "--output", str(fasta)])
