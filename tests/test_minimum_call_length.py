import contextlib
import hashlib
import json

import pytest
import torch

from talea import cli, inference
from talea.io import SequenceRecord


def record(name, length):
    sequence = "A" * length
    return SequenceRecord(name, sequence, hashlib.sha256(sequence.encode()).hexdigest())


@pytest.fixture
def encoder_stub(monkeypatch):
    """Replace the expensive encoder while retaining Talea's readouts and score calculation."""
    seen = []

    def load_models(predictor):
        predictor.esm_model = None
        predictor.probe_layer = None
        predictor.teacher_model, predictor.teacher_contract = inference._load_teacher(
            predictor.assets.teacher_model, predictor.assets.teacher_contract, predictor.device
        )

        def capture_batch(batch, *unused):
            seen.extend(r.record_id for r in batch)
            width = max(r.length for r in batch) + 2
            indices = [(torch.arange(1, r.length + 1), slice(1, r.length + 1)) for r in batch]
            weights = torch.ones(len(batch), 1, width, width) / width
            states = torch.zeros(len(batch), width, 1152)
            attentions = dict.fromkeys(predictor.expected_layers, weights)
            hidden = dict.fromkeys(set(predictor.expected_layers) | set(predictor.openness_layers), states)
            return (indices, attentions, hidden, None), None

        monkeypatch.setattr(inference, "_forward_pair", capture_batch)

    monkeypatch.setattr(inference.TaleaPredictor, "_load_models", load_models)
    monkeypatch.setattr(inference, "selective_capture", lambda *a, **k: contextlib.nullcontext())
    return seen


def test_all_short_cli_skips_model_and_audits_records(tmp_path, monkeypatch):
    def unexpected_model(**kwargs):
        pytest.fail("attempted to load a model for short proteins")

    monkeypatch.setattr(cli, "TaleaPredictor", unexpected_model)
    source = tmp_path / "proteins.faa"
    source.write_text(">z\n" + "A" * 20 + "*\n>a\nC\n")
    output = tmp_path / "out.jsonl"
    assert cli.main(["-i", str(source), "-o", str(output)]) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(r["record_id"], r["length"]) for r in rows] == [("a", 1), ("z", 20)]
    for r in rows:
        assert r["status"] == "below_minimum_length"
        assert r["talea_discovery_score"] is None
        assert r["talea_discovery_score_ungated"] is None
        assert r["wedge_call"] is None and r["repeat_regions"] == []
    audit = json.loads((tmp_path / "out.jsonl.audit.json").read_text())
    assert (audit["records"], audit["scored_records"], audit["below_minimum_length"]) == (2, 0, 2)
    assert audit["minimum_call_length"] == 21 and audit["device"] is None


@pytest.mark.parametrize("windowed", [False, True])
def test_mixed_cli_scores_only_eligible_sequences(tmp_path, monkeypatch, encoder_stub, windowed):
    lengths = [22, 1, 21, 20]
    options = []
    if windowed:
        # A small model context exercises the real window merger with two windows.
        contract = cli.load_deployment()
        contract["input_contract"]["maximum_residues"] = 28
        monkeypatch.setattr(cli, "load_deployment", lambda: contract)
        monkeypatch.setattr(inference.TaleaPredictor, "maximum_length", property(lambda self: 28))
        lengths.append(44)
        options = ["--long-proteins", "window", "--window-length", "22", "--window-stride", "22"]
    source = tmp_path / "proteins.faa"
    source.write_text("".join(f">p{n}\n{'A' * n}\n" for n in lengths))
    output = tmp_path / "out.jsonl"
    args = ["-i", str(source), "-o", str(output), "--device", "cpu", "--unit-placement", "comb",
            "--postprocess-mode", "thread", *options]
    assert cli.main(args) == 0
    rows = {r["record_id"]: r for r in map(json.loads, output.read_text().splitlines())}
    assert set(rows) == {f"p{n}" for n in lengths}
    assert sorted(encoder_stub) == sorted(["p21", "p22", *(["p44/1-22", "p44/23-44"] if windowed else [])])
    assert all(rows[f"p{n}"]["status"] == "below_minimum_length" for n in (1, 20))
    for n in lengths:
        row = rows[f"p{n}"]
        assert row["sequence_sha256"] == record(row["record_id"], n).sequence_sha256
        if n >= 21:
            assert all(isinstance(row[f], float) for f in ("talea_discovery_score", "talea_discovery_score_ungated"))
    audit = json.loads((tmp_path / "out.jsonl.audit.json").read_text())
    assert audit["records"] == len(lengths) and audit["scored_records"] == len(lengths) - 2
    if windowed:
        assert rows["p44"]["windowing"]["window_count"] == audit["scored_windows"] == 2

    # Running only the eligible proteins must give the same scores and calls.
    eligible = [record(f"p{n}", n) for n in lengths if n >= 21]
    with inference.TaleaPredictor(device="cpu", unit_placement="comb") as predictor:
        if windowed:
            controls = predictor.iter_windowed_predictions(
                eligible, window_length=22, window_stride=22, postprocess_mode="thread"
            )
        else:
            controls = predictor.iter_predictions(eligible, postprocess_mode="thread")
        for control in controls:
            observed = rows[control["record_id"]]
            for field in ("status", "talea_discovery_score", "talea_discovery_score_ungated",
                          "wedge_call", "repeat_regions"):
                assert observed[field] == control[field]


def test_python_api_validates_before_skipping(encoder_stub):
    with inference.TaleaPredictor(device="cpu") as predictor:
        tiny = record("tiny", 1)
        bad = SequenceRecord("bad", "X", hashlib.sha256(b"X").hexdigest())
        for records, reason in [([tiny, bad], "noncanonical"), ([tiny, tiny], "duplicate"),
                                ([SequenceRecord("bad", "C", "")], "checksum")]:
            with pytest.raises(ValueError, match=reason):
                next(predictor.iter_predictions(records))
        rows = list(predictor.iter_windowed_predictions([tiny], window_length=10, window_stride=5))
        assert rows[0]["status"] == "below_minimum_length"
        giant = record("giant", 2050)
        with pytest.raises(ValueError, match="at least 21"):
            list(predictor.iter_windowed_predictions([giant], window_length=20, window_stride=10))
        with pytest.raises(ValueError, match="duplicate"):
            list(predictor.iter_windowed_predictions([giant, record("giant", 1)]))
    assert not encoder_stub
