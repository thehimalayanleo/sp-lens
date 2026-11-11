#!/usr/bin/env python3
"""Turn a demo_keywords.py export into the compact file the SP-Lens Keywords page reads.

Keywords are a display step that never sees labels: word-like tokens only, tokens present in
the lists of >= 5% of prompts dropped, score = sum over layers of 1/rank * log(N / df)."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re

ARMS = ["J", "R", "logit", "JR+prev", "J-free"]
LAYER_ARMS = ["J", "R", "logit", "JR+prev"]
WORD = re.compile(r"^\s?[A-Za-zÀ-ɏЀ-ӿ][a-zß-ɏа-я]{2,}$|^\s?\d+$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("export")
    p.add_argument("out")
    args = p.parse_args()
    d = json.load(open(args.export))
    V = [s.replace("�", "▯") for s in d["vocab"]]
    items = d["items"]
    N = len(items)
    ok = lambda s: bool(WORD.match(s))
    df = {a: collections.Counter() for a in ARMS}
    for x in items:
        for a in ARMS:
            df[a].update({t for layer in x["layers"][a] for t, _ in layer})

    def keywords(x, a, n=10):
        score, hit, layers = collections.defaultdict(float), collections.defaultdict(int), collections.defaultdict(set)
        for li, layer in enumerate(x["layers"][a]):
            for rank, (t, h) in enumerate(layer, start=1):
                if not ok(V[t]) or df[a][t] / N >= 0.05:
                    continue
                k = V[t].strip().lower()
                score[k] += math.log(N / df[a][t]) / rank
                hit[k] |= h
                layers[k].add(li)
        return [[k, hit[k], len(layers[k])] for k in sorted(score, key=lambda k: -score[k])[:n]]

    found = lambda x, a: any(x["found"][a])
    scored = [x for x in items if x["found"]["J"]]
    stats = {}
    for a in ARMS:
        toks = [V[t] for x in items for layer in x["layers"][a] for t, _ in layer]
        stats[a] = {"found": sum(found(x, a) for x in scored) / len(scored),
                    "wordlike": sum(map(ok, toks)) / len(toks),
                    "kwhit": sum(any(h for _, h, _ in keywords(x, a)) for x in items) / N}
    stats["pairs"] = {b: {"sp_only": sum(found(x, "JR+prev") and not found(x, b) for x in scored),
                          "base_only": sum(found(x, b) and not found(x, "JR+prev") for x in scored)}
                      for b in ["J", "R", "logit"]}
    cats = sorted({x["cat"] for x in items})
    stats["catn"] = {c: sum(x["cat"] == c for x in items) for c in cats}
    stats["cats"] = {c: {a: sum(found(x, a) for x in scored if x["cat"] == c) /
                         max(1, sum(x["cat"] == c for x in scored)) for a in ARMS} for c in cats}
    stats["n"] = len(scored)

    vocab, vid = [], {}

    def v(t):
        s = V[t]
        if s not in vid:
            vid[s] = len(vocab)
            vocab.append(s)
        return vid[s]

    out = [{"c": x["cat"], "p": x["prompt"], "r": x["readout"].replace("�", "▯"), "i": x["inter"],
            "f": {a: x["found"][a] for a in ARMS}, "k": {a: keywords(x, a) for a in ARMS},
            "L": {a: [[[v(t), h] for t, h in layer] for layer in x["layers"][a]] for a in LAYER_ARMS}}
           for x in items]
    looped = any(s for s, _ in d["labels"])
    labels = [f"{s}.{l}" if looped else str(l) for s, l in d["labels"]]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": d["model"], "layers": labels, "vocab": vocab, "stats": stats, "items": out}, f,
                  separators=(",", ":"), ensure_ascii=False)
    print(args.out, N, {a: round(stats[a]["found"], 3) for a in ARMS}, stats["pairs"])


if __name__ == "__main__":
    main()
