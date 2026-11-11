"""Find decoder layers, read their outputs and record every layer call."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import Tensor, nn


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for outer, inner in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        parent = getattr(model, outer, None)
        layers = getattr(parent, inner, None) if parent is not None else None
        if layers is not None:
            return layers
    raise TypeError("unsupported model: could not find decoder layers")


def final_norm(model: nn.Module) -> nn.Module:
    for outer, inner in (("model", "norm"), ("transformer", "ln_f"), ("gpt_neox", "final_layer_norm")):
        parent = getattr(model, outer, None)
        norm = getattr(parent, inner, None) if parent is not None else None
        if norm is not None:
            return norm
    raise TypeError("unsupported model: could not find the final norm")


def hidden_from_layer_output(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"unsupported decoder-layer output type: {type(output)!r}")


@contextmanager
def record_calls(model: nn.Module) -> Iterator[list[tuple[int, Tensor]]]:
    """Record (layer index, hidden) for every decoder-layer call in order.

    Looped models such as Ouro run each layer once per loop step, so one
    layer can appear several times.
    """
    calls: list[tuple[int, Tensor]] = []
    handles = [
        layer.register_forward_hook(
            lambda _m, _a, out, i=i: calls.append((i, hidden_from_layer_output(out)))
        )
        for i, layer in enumerate(decoder_layers(model))
    ]
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def call_labels(n_calls: int, n_layers: int) -> list[tuple[int, int]]:
    """(loop step, layer) for each call."""
    return [(k // n_layers, k % n_layers) for k in range(n_calls)]
