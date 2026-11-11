"""Read what a model holds at each layer with SP-Lens and its component lenses.

Per source layer call l and readout position p, each component lens gives
log-probabilities over the vocabulary:

  logit : final norm + unembedding of h_l[p]                    (no fitting)
  prev  : logit lens of h_l at earlier positions, max over [4, p) (no fitting)
  j, r  : fitted J-lens / R-lens transport of h_l[p], then unembedding

SP-Lens cleans each list and fuses them (the SPLENS-V4 "JR+prev" arm):
  1. PMI correction: subtract 0.5 x the lens's mean log-prob on calibration text.
  2. Distinct: keep the first token per normalized surface form, drop prompt
     tokens, keep 10.
  3. Fuse: score = sum over lenses of weight / rank, minus
     0.35 x min(times shown at earlier layers, 3). Show the top 10.

Without fitted lenses, the J-free arm (logit 1.0, prev 0.5) is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .fit import corpus
from .hooks import call_labels, decoder_layers, final_norm, record_calls

JR_PREV = {"r": 1.0, "j": 0.25, "logit": 0.25, "prev": 0.5}
J_FREE = {"logit": 1.0, "prev": 0.5}


def norm_text(s: str) -> str:
    return re.sub(r"[^\w]", "", s.strip().lower())


@dataclass
class LayerReading:
    label: tuple[int, int]  # (loop step, layer)
    sp: list[str]
    lenses: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Reading:
    prompt: str
    position: int
    token: str
    layers: list[LayerReading]

    def table(self, lens: str = "sp", k: int = 10) -> str:
        rows = []
        for layer in self.layers:
            toks = layer.sp if lens == "sp" else layer.lenses[lens]
            step, idx = layer.label
            name = f"L{idx}" if step == 0 else f"s{step}.L{idx}"
            rows.append(f"{name:>8}  " + " | ".join(t.strip() or repr(t) for t in toks[:k]))
        return "\n".join(rows)


class SPLens:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        lens_dir: str | Path | None = None,
        *,
        weights: dict[str, float] | None = None,
        pmi: float = 0.5,
        novelty: float = 0.35,
        pool: int = 200,
        top: int = 10,
        skip_first: int = 4,
    ):
        self.model, self.tok = model.eval(), tokenizer
        self.device = next(model.parameters()).device
        self.n_layers = len(decoder_layers(model))
        self.maps: dict[str, torch.Tensor] = {}
        self.labels: list[tuple[int, int]] | None = None
        if lens_dir is not None:
            for rule in "jr":
                path = Path(lens_dir) / f"{rule}-lens" / "lens.pt"
                if path.exists():
                    payload = torch.load(path, weights_only=False)
                    self.maps[rule] = payload["J"].to(self.device).float()
                    self.labels = [tuple(x) for x in payload["source_labels"]]
        if weights is None:
            weights = JR_PREV if set(self.maps) == {"j", "r"} else J_FREE
        missing = {w for w in weights if w in "jr"} - set(self.maps)
        if missing:
            raise ValueError(f"weights use {sorted(missing)} but no fitted lens was found")
        self.weights = weights
        self.pmi, self.novelty, self.pool, self.top, self.skip_first = pmi, novelty, pool, top, skip_first
        self.base: dict[str, torch.Tensor] | None = None
        vocab = model.get_output_embeddings().weight.shape[0]
        self.surface = [norm_text(tokenizer.decode([t])) for t in range(vocab)]

    def _readout(self, x: torch.Tensor) -> torch.Tensor:
        head = self.model.get_output_embeddings()
        return head(final_norm(self.model)(x.to(self.model.dtype))).float().log_softmax(-1)

    def _lens(self, rule: str, x: torch.Tensor) -> torch.Tensor:
        # x [L, ..., d], maps [L, d, d]
        return self._readout(torch.einsum("l...d,lkd->l...k", x.float(), self.maps[rule]))

    @torch.inference_mode()
    def _states(self, ids: torch.Tensor) -> torch.Tensor:
        """Hidden states [L, T, d] of the source layer calls."""
        with record_calls(self.model) as calls:
            self.model(input_ids=ids, use_cache=False)
        if self.labels is None:
            self.labels = call_labels(len(calls) - 2, self.n_layers)
        index = [s * self.n_layers + l for s, l in self.labels]
        return torch.stack([calls[k][1][0].float() for k in index])

    @torch.inference_mode()
    def calibrate(self, prompts: int = 25, length: int = 128, corpus_file: str | None = None) -> None:
        """Mean log-prob per lens over calibration text, for the PMI correction."""
        names = [n for n in self.weights if n != "prev"]
        base, count = {}, 0
        for ids in corpus(self.tok, prompts, length, corpus_file):
            hs = self._states(ids[None].to(self.device))[:, self.skip_first:]
            for n in names:
                lp = self._readout(hs) if n == "logit" else self._lens(n, hs)
                base[n] = base.get(n, 0) + lp.sum(1)
            count += hs.shape[1]
        self.base = {n: v / count for n, v in base.items()}
        if "prev" in self.weights:
            self.base["prev"] = self.base.get("logit", self._zero())

    def _zero(self) -> torch.Tensor:
        return torch.zeros(len(self.labels), len(self.surface), device=self.device)

    def _distinct(self, scores: torch.Tensor, banned: set[int]) -> list[list[int]]:
        out = []
        for row in scores.topk(self.pool, dim=-1).indices.tolist():
            chosen, seen = [], set()
            for t in row:
                if self.surface[t] in seen or t in banned:
                    continue
                seen.add(self.surface[t])
                chosen.append(t)
                if len(chosen) == self.top:
                    break
            out.append(chosen)
        return out

    @torch.inference_mode()
    def read(self, prompt: str, position: int = -1, lenses: tuple[str, ...] = ()) -> Reading:
        """SP-Lens tokens per layer at `position`; also raw top lists for `lenses`."""
        if self.base is None:
            raise RuntimeError("call calibrate() first")
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)
        pos = position % ids.shape[1]
        hs = self._states(ids)
        h = hs[:, pos]
        lp = {}
        for n in set(self.weights) | set(lenses):
            if n == "logit":
                lp[n] = self._readout(h)
            elif n == "prev":
                earlier = hs[:, self.skip_first:pos] if pos > self.skip_first else hs[:, :pos]
                lp[n] = self._readout(earlier).amax(1) if earlier.shape[1] else self._readout(h)
            else:
                lp[n] = self._lens(n, h)
        banned = set(ids[0].tolist())
        lists = {n: self._distinct(lp[n] - self.pmi * self.base[n], banned) for n in self.weights}
        shown: dict[int, int] = {}
        layers = []
        for l, label in enumerate(self.labels):
            score: dict[int, float] = {}
            for n, w in self.weights.items():
                for rank, t in enumerate(lists[n][l], start=1):
                    score[t] = score.get(t, 0.0) + w / rank
            for t in score:
                score[t] -= self.novelty * min(shown.get(t, 0), 3)
            best = sorted(score, key=lambda t: (-score[t], t))[: self.top]
            for t in best:
                shown[t] = shown.get(t, 0) + 1
            raw = {n: [self.tok.decode([t]) for t in lp[n][l].topk(self.top).indices.tolist()] for n in lenses}
            layers.append(LayerReading(label, [self.tok.decode([t]) for t in best], raw))
        return Reading(prompt, pos, self.tok.decode([ids[0, pos].item()]), layers)
