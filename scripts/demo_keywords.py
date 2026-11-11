#!/usr/bin/env python3
"""Keyword lists for the SP-Lens demo page (not an experiment).

Confirmation items only, and only the arms SPLENS-V4 already reported (plain J, plain R,
logit lens, JR+prev, J-free), so nothing here touches the SPLENS-POS development split or
its control arms. Cleaning is exactly experiment 126 (PMI 0.5, distinct, prompt tokens
removed, novelty 0.35)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from splens.external_workspace_eval import intermediate_target_sets, load_workspace_evals, readout_position

_here = Path(__file__).resolve().parent.parent / "experiments"
_spec = importlib.util.spec_from_file_location("v4", _here / "120_v4_lens_rows.py")
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)
_spec = importlib.util.spec_from_file_location("s126", _here / "126_signals_beyond_jr.py")
s126 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s126)


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lens-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--jfree", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--calib-prompts", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    device = torch.device(args.device)
    tok, model = v4.load(args.model, device)
    n_layers = len(model.model.layers)
    lens = {r: torch.load(Path(args.lens_dir) / f"{r}-lens" / "lens.pt", weights_only=False) for r in "jr"}
    labels = [tuple(x) for x in lens["j"]["source_labels"]]
    maps = {r: lens[r]["J"].to(device).float() for r in "jr"}
    V = model.lm_head.weight.shape[0]
    surface = [s126.norm_text(tok.decode([t])) for t in range(V)]

    def states(ids):
        with v4.record_calls(model) as calls:
            model(input_ids=ids, use_cache=False)
        lab = v4.call_labels(len(calls), n_layers)
        hs, prev = [], []
        for label in labels:
            k = lab.index(label)
            hs.append(calls[k][1][0].float())
            prev.append(calls[k - 1][1][0].float() if k > 0 else torch.zeros_like(calls[k][1][0].float()))
        return torch.stack(hs), torch.stack(prev)

    lp_logit = lambda x: v4.readout(model, x).log_softmax(-1)
    lp_lens = lambda r, x: v4.readout(model, torch.einsum("ld,lkd->lk", x, maps[r])).log_softmax(-1)

    base = {k: torch.zeros(len(labels), V, device=device) for k in ("logit", "delta", "r", "j")}
    count = 0
    for ids in v4.corpus(tok, args.calib_prompts, 128, None):
        hs, pv = states(ids[None].to(device))
        for l in range(len(labels)):
            h, d = hs[l, 4:], hs[l, 4:] - pv[l, 4:]
            base["logit"][l] += lp_logit(h).sum(0)
            base["delta"][l] += lp_logit(d).sum(0)
            for r in "jr":
                base[r][l] += v4.readout(model, h @ maps[r][l].T).log_softmax(-1).sum(0)
        count += ids.numel() - 4
    for k in base:
        base[k] /= count
    base["prev"] = base["logit"]

    arms = {"JR+prev": {"r": 1.0, "j": 0.25, "logit": 0.25, "prev": 0.5},
            "J-free": {s: (1.0 if s == "logit" else 0.5) for s in args.jfree.split("+")}}
    items = [it for it in load_workspace_evals(args.data_root) if it.split == "confirmation"]
    if args.limit:
        items = items[: args.limit]
    vocab, vid = [], {}

    def v(t):
        s = tok.decode([t])
        if s not in vid:
            vid[s] = len(vocab)
            vocab.append(s)
        return vid[s]

    out = []
    for n, item in enumerate(items):
        ids = tok(item.prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        pos = readout_position(tok, item) % ids.shape[1]
        hs, pv = states(ids)
        h = hs[:, pos]
        lp = {"logit": lp_logit(h), "delta": lp_logit(h - pv[:, pos]), "r": lp_lens("r", h), "j": lp_lens("j", h)}
        earlier = hs[:, 4:pos] if pos > 4 else hs[:, :pos]
        lp["prev"] = lp_logit(earlier).amax(dim=1) if earlier.shape[1] else lp["logit"]
        targets = intermediate_target_sets(tok, item.category, item.intermediates)
        allt = set().union(*targets) if targets else set()
        banned = set(ids[0].tolist())
        lists = {}
        for name in {s for w in arms.values() for s in w}:
            cand = (lp[name] - 0.5 * base[name]).topk(200, dim=-1).indices.tolist()
            rows = []
            for row in cand:
                chosen, seen = [], set()
                for t in row:
                    if surface[t] in seen or t in banned:
                        continue
                    seen.add(surface[t]); chosen.append(t)
                    if len(chosen) == 10:
                        break
                rows.append(chosen)
            lists[name] = rows
        sups = {"J": lp["j"].topk(10, dim=-1).indices.tolist(), "R": lp["r"].topk(10, dim=-1).indices.tolist(),
                "logit": lp["logit"].topk(10, dim=-1).indices.tolist()}
        for a, weights in arms.items():
            shown, per = {}, []
            for l in range(len(labels)):
                sc = {}
                for name, w in weights.items():
                    for rank, t in enumerate(lists[name][l], start=1):
                        sc[t] = sc.get(t, 0.0) + w / rank
                for t in sc:
                    sc[t] -= 0.35 * min(shown.get(t, 0), 3)
                top = sorted(sc, key=lambda t: (-sc[t], t))[:10]
                for t in top:
                    shown[t] = shown.get(t, 0) + 1
                per.append(top)
            sups[a] = per
        rec = {"cat": item.category, "name": item.name, "prompt": item.prompt,
               "readout": tok.decode(ids[0, pos]), "inter": [str(x) for x in item.intermediates],
               "layers": {}, "found": {}}
        for a, per in sups.items():
            rec["layers"][a] = [[[v(t), int(t in allt)] for t in layer] for layer in per]
            shown = {t for layer in per for t in layer}
            rec["found"][a] = [bool(ts & shown) for ts in targets]
        out.append(rec)
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(items)}", flush=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "labels": [list(x) for x in labels],
                                          "vocab": vocab, "items": out}, separators=(",", ":")))
    print("wrote", args.out, len(out), "items")


if __name__ == "__main__":
    main()
