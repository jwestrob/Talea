"""CUDA replay of the original per-record hidden readout operations for repeated lengths."""

from collections import Counter

import torch


class HiddenReadoutGraphs:
    """Own one repeatedly used sequence length in a sorted batch stream."""

    def __init__(self, predictor, records, minimum_uses):
        self.predictor = predictor
        counts = Counter(record.length for record in records)
        self.eligible = {length for length, count in counts.items() if count >= minimum_uses}
        self.layers = tuple(sorted(set(predictor.expected_layers) | set(predictor.openness_layers)))
        self.length = None
        self.graph = None
        self.side = torch.cuda.Stream(device=predictor.device)
        self.stats = {"captures": 0, "replays": 0}

    def _capture(self, sources):
        # The reference casts each layer into an independent, aligned FP32 allocation.
        self.inputs = {layer: torch.empty_like(source, dtype=torch.float32, memory_format=torch.contiguous_format)
                       for layer, source in zip(self.layers, sources, strict=True)}
        self.destinations = list(self.inputs.values())
        def forward():
            outputs = self.predictor._hidden_device_outputs(self.inputs.__getitem__)
            return torch.cat([value.reshape(-1) for value in outputs.values()]), outputs
        torch._foreach_copy_(self.destinations, sources)
        # Initialize cuBLAS and the reduction kernels for this exact shape.
        forward()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            self.output, outputs = forward()
        finally:
            graph.capture_end()
        self.layout = tuple((name, value.shape) for name, value in outputs.items())
        self.graph = graph
        self.stats["captures"] += 1

    def batch(self, hidden_states, indices_by_record, stream):
        prepared = []
        device = self.predictor.device
        stream = stream if stream is not None else torch.cuda.current_stream(device)
        execution = self.side if stream == torch.cuda.default_stream(device) else stream
        if execution != stream:
            execution.wait_stream(stream)
        with torch.cuda.device(device), torch.cuda.stream(execution):
            for index, (indices, span) in enumerate(indices_by_record):
                length = len(indices)
                if length not in self.eligible or span is None:
                    prepared.append(None)
                    continue
                sources = [(hidden_states[layer][index] if hidden_states[layer].ndim == 2 else
                            hidden_states[layer][index, span]) for layer in self.layers]
                if self.length != length:
                    # Each capture has an independent pool. The previous length's
                    # graph, inputs and output can be released after enqueued use.
                    self._capture(sources)
                    self.length = length
                torch._foreach_copy_(self.destinations, sources)
                self.graph.replay()
                # Each record readout consumes its own copy on the shared readout
                # stream; later graph launches may overwrite the captured output.
                prepared.append((self.output.clone(), self.layout))
                self.stats["replays"] += 1
        if execution != stream:
            stream.wait_stream(execution)
            for row in prepared:
                if row is not None:
                    row[0].record_stream(stream)
        return prepared
