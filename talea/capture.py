"""Selective capture of ESM++ attention and hidden states.

Talea consumes two attention maps and two pooled hidden states from a 36-layer
ESM++ forward pass.  Requesting them through ``output_attentions=True`` /
``output_hidden_states=True`` is expensive for two independent reasons:

1. ``MultiHeadAttention._attn`` short-circuits to ``_manual_attn`` whenever
   ``output_attentions`` is set, so *every* layer abandons the configured SDPA
   backend and materializes a full ``B x H x T x T`` score matrix.
2. ``TransformerStack.forward`` accumulates every layer's weights into a tuple
   that stays live for the whole forward pass.  At L=2046 with 18 heads in
   fp16 that is 144 MB per layer, 5.06 GB over 36 layers, of which Talea uses
   288 MB.

This module captures exactly what Talea needs with per-module patches, leaving
the rest of the stack free to keep its fast path and retain nothing.

Two modes are provided because they differ in numerical status:

``manual``
    Every block keeps the ``_manual_attn`` path it takes today, so the residual
    stream is bit-identical to the unpatched ``output_attentions=True`` run.
    Only the retention is removed.  This is the default.

``sdpa``
    Only the two captured blocks use ``_manual_attn``; the other 34 dispatch
    normally (SDPA).  Faster, but SDPA and the manual path accumulate
    differently, so the residual stream drifts slightly and downstream scores
    are not bit-identical.  Opt in only with a concordance run in hand.

Index conventions, verified empirically in tests/test_concordance.py rather
than assumed:

* ``attentions[i]`` is block ``i``'s weights, so attention layers 13 and 35 are
  blocks 13 and 35.
* ``hidden_states`` does NOT begin with the embedding for this model (the
  prepend at ``TransformerStack.forward`` is gated on ``esmfold2_hidden_states``,
  which Talea never sets).  ``hidden_states[i]`` is therefore block ``i``'s
  output and ``hidden_states[36]`` is the final ``norm(x)``.  Talea reads
  ``hidden_states[layer + 1]``, so layer 13 resolves to block 14's output and
  layer 35 resolves to ``last_hidden_state``.  That off-by-one is part of the
  frozen feature contract and is reproduced here, not corrected.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import torch


CAPTURE_MODES = ("manual", "sdpa")


def _blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError("ESM++ model exposes no .transformer stack")
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise AttributeError("ESM++ transformer stack exposes no .blocks")
    return blocks


def resolve_hidden_state_sources(
    transformer_layers: Sequence[int], n_blocks: int
) -> dict[int, int | None]:
    """Map each requested layer to the block whose output Talea pools.

    ``None`` means the value is ``last_hidden_state`` rather than a block
    output, which is what ``hidden_states[n_blocks]`` resolves to.
    """
    sources: dict[int, int | None] = {}
    for layer in transformer_layers:
        index = int(layer) + 1
        if index < 0 or index > n_blocks:
            raise ValueError(
                f"hidden-state index {index} outside the {n_blocks}-block stack"
            )
        sources[int(layer)] = None if index == n_blocks else index
    return sources


class SelectiveCapture:
    """Capture chosen attention maps and hidden states from one forward pass."""

    def __init__(
        self,
        model: torch.nn.Module,
        attention_layers: Sequence[int],
        transformer_layers: Sequence[int],
        mode: str = "manual",
    ):
        if mode not in CAPTURE_MODES:
            raise ValueError(f"unsupported capture mode: {mode}")
        self.model = model
        self.mode = mode
        self.attention_layers = tuple(int(value) for value in attention_layers)
        self.blocks = _blocks(model)
        self.n_blocks = len(self.blocks)
        for layer in self.attention_layers:
            if not 0 <= layer < self.n_blocks:
                raise ValueError(
                    f"attention layer {layer} outside the {self.n_blocks}-block stack"
                )
        self.hidden_sources = resolve_hidden_state_sources(
            transformer_layers, self.n_blocks
        )
        self.attentions: dict[int, torch.Tensor] = {}
        self.hidden_states: dict[int, torch.Tensor] = {}
        self._handles: list = []
        self._patched: list[tuple[torch.nn.Module, object]] = []

    # ------------------------------------------------------------------ setup
    def _patch_attention(self, block_index: int, block: torch.nn.Module) -> None:
        attention = block.attn
        original = attention._attn
        capture = block_index in self.attention_layers
        sink = self.attentions

        def patched(
            query_BHLD,
            key_BHLD,
            value_BHLD,
            attention_mask_2d=None,
            attention_mask_4d=None,
            flex_block_mask=None,
            output_attentions=False,
            output_s_max=False,
        ):
            # Force the same manual kernel the unpatched output_attentions=True
            # run uses, so the residual stream is unchanged.
            output, weights, s_max = attention._manual_attn(
                query_BHLD,
                key_BHLD,
                value_BHLD,
                attention_mask_4d,
                output_s_max,
            )
            if capture:
                sink[block_index] = weights.detach()
            # Returning None drops the reference immediately for uncaptured
            # blocks; the stack only stores weights when it built a tuple, and
            # we always call it with output_attentions=False.
            return output, (weights if capture else None), s_max

        attention._attn = patched
        self._patched.append((attention, original))

    def _patch_capture_only(self, block_index: int, block: torch.nn.Module) -> None:
        attention = block.attn
        original = attention._attn
        sink = self.attentions

        def patched(
            query_BHLD,
            key_BHLD,
            value_BHLD,
            attention_mask_2d=None,
            attention_mask_4d=None,
            flex_block_mask=None,
            output_attentions=False,
            output_s_max=False,
        ):
            output, weights, s_max = attention._manual_attn(
                query_BHLD,
                key_BHLD,
                value_BHLD,
                attention_mask_4d,
                output_s_max,
            )
            sink[block_index] = weights.detach()
            return output, None, s_max

        attention._attn = patched
        self._patched.append((attention, original))

    def _hook_hidden(self, block_index: int, layer: int) -> None:
        def hook(module, args, output):
            # A block returns (x, attn_weights, s_max); hidden_states stores x.
            self.hidden_states[layer] = output[0].detach()

        self._handles.append(self.blocks[block_index].register_forward_hook(hook))

    def install(self) -> None:
        for index, block in enumerate(self.blocks):
            if self.mode == "manual":
                self._patch_attention(index, block)
            elif index in self.attention_layers:
                self._patch_capture_only(index, block)
        for layer, block_index in self.hidden_sources.items():
            if block_index is not None:
                self._hook_hidden(block_index, layer)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attention, original in self._patched:
            attention._attn = original
        self._patched.clear()

    def reset(self) -> None:
        self.attentions.clear()
        self.hidden_states.clear()

    # ------------------------------------------------------------------ output
    def finalize(self, last_hidden_state: torch.Tensor) -> None:
        """Supply ``last_hidden_state`` for any layer that resolves to it."""
        for layer, block_index in self.hidden_sources.items():
            if block_index is None:
                self.hidden_states[layer] = last_hidden_state

    def require(self) -> None:
        missing_attention = [
            layer for layer in self.attention_layers if layer not in self.attentions
        ]
        if missing_attention:
            raise RuntimeError(f"attention capture missed layers: {missing_attention}")
        missing_hidden = [
            layer for layer in self.hidden_sources if layer not in self.hidden_states
        ]
        if missing_hidden:
            raise RuntimeError(f"hidden-state capture missed layers: {missing_hidden}")


@contextmanager
def selective_capture(
    model: torch.nn.Module,
    attention_layers: Sequence[int],
    transformer_layers: Sequence[int],
    mode: str = "manual",
):
    capture = SelectiveCapture(model, attention_layers, transformer_layers, mode=mode)
    capture.install()
    try:
        yield capture
    finally:
        capture.remove()
