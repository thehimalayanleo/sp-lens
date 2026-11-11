"""Build site/data/landing.json, the small file the landing page reads.

For each demo export it keeps the per-layer hit grid (did the lens's top 10 hold the
labelled concept at that layer) for SP-Lens and the logit lens, the headline rates,
and a few showcase prompts with their decoded per-layer token lists.

    python3 scripts/landing_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("smollm3_3b", "SmolLM3-3B"), ("ouro_1p4b", "Ouro-1.4B"), ("ouro_2p6b", "Ouro-2.6B")]
LENSES = ["JR+prev", "logit", "J", "R"]
PER_CAT = 2  # showcase prompts per category


def hit_row(layers):
    return "".join("1" if any(h for _, h in toks) else "0" for toks in layers)


def build(name, label):
    d = json.loads((ROOT / "demo/data" / f"{name}.json").read_text())
    vocab = d["vocab"]
    items = d["items"]
    grid = {lens: [hit_row(it["L"][lens]) for it in items] for lens in ("JR+prev", "logit")}
    # Showcase: SP-Lens finds the concept and the logit lens never does. Poetry is left
    # out because every lens is near zero there.
    picks, per = [], {}
    for it in items:
        c = it["c"]
        if c == "poetry" or per.get(c, 0) >= PER_CAT:
            continue
        if any(it["f"]["JR+prev"]) and not any(it["f"]["logit"]):
            per[c] = per.get(c, 0) + 1
            picks.append({
                "c": c, "p": it["p"], "i": it["i"],
                "L": {lens: [[[vocab[t], h] for t, h in toks] for toks in it["L"][lens]] for lens in LENSES},
            })
    return {
        "key": name, "label": label, "model": d["model"], "layers": d["layers"],
        "cats": [it["c"] for it in items],
        "rate": {lens: d["stats"][lens]["found"] for lens in LENSES},
        "grid": grid, "show": picks,
    }


out = {"n": None, "models": [build(n, l) for n, l in MODELS]}
out["n"] = len(out["models"][0]["cats"])
dst = ROOT / "site/data/landing.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print(dst, f"{dst.stat().st_size / 1e3:.0f} kB", [len(m["show"]) for m in out["models"]])
