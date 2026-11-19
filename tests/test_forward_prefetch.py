"""Exercise capture ownership and one-batch scheduling without an accelerator."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
from types import SimpleNamespace

import pytest

from talea.capture import SelectiveCapture
from talea.cli import build_parser
from talea.inference import _forward_pair, _handoff_capture, _prefetch_memory_required


def test_prefetch_is_opt_in():
    parser = build_parser()
    paths = ["-i", "sample.faa", "-o", "predictions.jsonl"]
    assert not parser.parse_args(paths).prefetch_forward
    assert parser.parse_args([*paths, "--prefetch-forward"]).prefetch_forward


def test_forward_overlaps_readout_and_retains_original_batches_in_order():
    entered = Event()
    release = Event()
    batches = [("a", "b"), ("c",), ("d", "e", "f")]
    calls = []

    def forward(batch):
        calls.append((batch, current_thread().ident))
        if batch is batches[1]:
            entered.set()
            assert release.wait(3)
        return tuple("capture-" + record for record in batch)

    with ThreadPoolExecutor(1) as producer:
        first, ahead = _forward_pair(batches[0], batches[1], None, forward, producer, lambda *_: True)
        try:
            assert entered.wait(3)
            assert first == ("capture-a", "capture-b")
            assert not ahead.done()
            assert [b for b, _ in calls] == batches[:2]
        finally:
            release.set()
        # A denied boundary makes the third forward synchronous after readout.
        second, ahead = _forward_pair(batches[1], batches[2], ahead, forward, producer, lambda *_: False)
        assert second == ("capture-c",) and ahead is None
        third, ahead = _forward_pair(batches[2], None, ahead, forward, producer, lambda *_: True)
        assert third == ("capture-d", "capture-e", "capture-f") and ahead is None
    assert [b for b, _ in calls] == batches
    assert calls[0][1] == calls[2][1] != calls[1][1]


def test_lookahead_error_reaches_consumer_before_another_batch_starts():
    calls = []

    def forward(batch):
        calls.append(batch)
        if batch == "bad":
            raise ValueError("forward failed")
        return batch

    with ThreadPoolExecutor(1) as producer:
        _, ahead = _forward_pair("first", "bad", None, forward, producer, lambda *_: True)
        with pytest.raises(ValueError, match="forward failed"):
            _forward_pair("bad", "last", ahead, forward, producer, lambda *_: True)
    assert calls == ["first", "bad"]


def test_captured_dictionaries_outlive_sink_reuse():
    capture = object.__new__(SelectiveCapture)
    capture.attention_layers = (13,)
    capture.hidden_sources = {35: None}
    capture.attentions, capture.hidden_states = {13: "old map"}, {35: "old state"}
    original_sink = capture.attentions
    captured = capture.take()
    assert not capture.attentions and not capture.hidden_states
    original_sink[13] = "new map"
    capture.finalize("new state")
    assert captured == ({13: "old map"}, {35: "old state"})
    assert capture.take() == ({13: "new map"}, {35: "new state"})


def test_cuda_consumer_waits_for_event_and_registers_every_storage():
    events = []
    consumer = SimpleNamespace(wait_event=lambda event: events.append(event))
    def tensor(label):
        return SimpleNamespace(record_stream=lambda stream: events.append((label, stream)))
    _handoff_capture(consumer, "producer ready", {13: tensor("attention")},
                     {35: tensor("hidden")}, [(tensor("indices"), None)])
    assert events == ["producer ready", ("attention", consumer), ("hidden", consumer), ("indices", consumer)]


def test_memory_guard_accounts_for_length_batch_size_and_readout_concurrency():
    current = [SimpleNamespace(length=200)]
    following = [SimpleNamespace(length=250)]
    attention = SimpleNamespace(shape=(1, 18, 202, 202), element_size=lambda: 2)
    hidden = SimpleNamespace(shape=(1, 202, 960), element_size=lambda: 2)
    single = _prefetch_memory_required(current, following, attention, hidden, 1)
    assert _prefetch_memory_required(current, following * 3, attention, hidden, 1) > single
    assert _prefetch_memory_required(current, following, attention, hidden, 3) > single
    assert _prefetch_memory_required(current, [SimpleNamespace(length=1200)], attention, hidden, 1) > single
