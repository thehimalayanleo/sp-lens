"""Fit J-lens and R-lens transports for any decoder-only Hugging Face model.

Each transport is the corpus-averaged Jacobian from a source layer call to the
second-to-last layer call, averaged over position pairs t <= t'. The R-lens
uses forward-equivalent backward rules (see r_lens.py), so the model's logits
do not change but the Jacobian does. This is the SPLENS-V4 recipe.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from .hooks import call_labels, decoder_layers, record_calls
from .r_lens import gated_r_lens_rules


def corpus(tokenizer, n: int, length: int, path: str | None = None) -> list[torch.Tensor]:
    """n prompts of exactly `length` tokens. Default text is NeelNanda/pile-10k."""
    if path:
        texts = Path(path).read_text().split("\n\n")
    else:
        from datasets import load_dataset
        texts = (row["text"] for row in load_dataset("NeelNanda/pile-10k", split="train"))
    out = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        if ids.numel() >= length:
            out.append(ids[:length])
        if len(out) == n:
            return out
    raise RuntimeError(f"corpus gave {len(out)} of {n} prompts of length {length}")


def source_indices(n_calls: int, n_layers: int, stride: int) -> list[int]:
    """Every call before the target call (n_calls - 2), every `stride`-th layer."""
    return [k for k in range(n_calls - 2) if (k % n_layers) % stride == 0]


def causal_jacobians(target, sources, skip_first: int, chunk: int) -> list[torch.Tensor]:
    """Mean over pairs t <= t' (both >= skip_first) of d target[t'] / d source[t]."""
    _, T, d = target.shape
    n = T - skip_first
    pairs = n * (n + 1) / 2
    rows: list[list[torch.Tensor]] = [[] for _ in sources]
    for start in range(0, d, chunk):
        stop = min(start + chunk, d)
        cot = torch.zeros(stop - start, *target.shape, device=target.device, dtype=target.dtype)
        cot[torch.arange(stop - start), 0, skip_first:, torch.arange(start, stop)] = 1
        grads = torch.autograd.grad(
            target, tuple(sources), grad_outputs=cot,
            retain_graph=stop < d, is_grads_batched=True,
        )
        for out, g in zip(rows, grads):
            out.append(g[:, 0, skip_first:].float().sum(dim=1) / pairs)
    return [torch.cat(r, dim=0) for r in rows]


def check_r_rules(model: nn.Module, ids: torch.Tensor) -> float:
    """Max change in logits when the R rules are on. Must be 0."""
    with torch.no_grad():
        plain = model(input_ids=ids, use_cache=False).logits.float()
        with gated_r_lens_rules(model):
            ruled = model(input_ids=ids, use_cache=False).logits.float()
    return float((plain - ruled).abs().max())


def fit_lenses(
    model: nn.Module,
    tokenizer,
    out: str | Path,
    *,
    rules: tuple[str, ...] = ("j", "r"),
    prompts: int = 25,
    length: int = 128,
    skip_first: int = 4,
    stride: int = 1,
    chunk: int = 32,
    corpus_file: str | None = None,
    log=print,
) -> Path:
    """Fit and save `<out>/<rule>-lens/lens.pt` for each rule.

    The model must be loaded with attn_implementation="eager", because the
    batched backward needs a plain attention graph.
    """
    device = next(model.parameters()).device
    n_layers = len(decoder_layers(model))
    texts = corpus(tokenizer, prompts, length, corpus_file)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    delta = check_r_rules(model, texts[0][None].to(device)) if "r" in rules else 0.0
    if delta > 0:
        raise RuntimeError(f"R rules changed forward logits by {delta}; refusing to fit")

    for rule in rules:
        started = time.time()
        means, labels = None, None
        with gated_r_lens_rules(model) if rule == "r" else nullcontext():
            for p, ids in enumerate(texts):
                model.zero_grad(set_to_none=True)
                with record_calls(model) as calls:
                    model(input_ids=ids[None].to(device), use_cache=False)
                index = source_indices(len(calls), n_layers, stride)
                labels = [call_labels(len(calls), n_layers)[k] for k in index]
                ops = causal_jacobians(calls[-2][1], [calls[k][1] for k in index], skip_first, chunk)
                if means is None:
                    means = [op.clone() for op in ops]
                else:
                    for m, op in zip(means, ops):
                        m.add_((op - m) / (p + 1))
                del calls, ops
                log(f"{rule}-lens {p + 1}/{len(texts)}")
        payload = {
            "J": torch.stack(means).to(torch.float16).cpu(),
            "source_labels": labels,
            "d_model": int(means[0].shape[0]),
            "n_prompts": len(texts),
            "provenance": {
                "model_id": getattr(model.config, "_name_or_path", None),
                "model_commit": getattr(model.config, "_commit_hash", None),
                "rule": rule,
                "target": "second-to-last decoder-layer call",
                "skip_first": skip_first,
                "length": length,
                "corpus": corpus_file or "NeelNanda/pile-10k",
                "r_forward_max_abs_delta": delta,
                "fit_seconds": time.time() - started,
            },
        }
        (out / f"{rule}-lens").mkdir(exist_ok=True)
        torch.save(payload, out / f"{rule}-lens" / "lens.pt")
    return out
