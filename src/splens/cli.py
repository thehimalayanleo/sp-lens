"""Command line: `splens fit` and `splens read`."""

from __future__ import annotations

import argparse

from . import SPLens, fit_lenses, load_model


def main() -> None:
    p = argparse.ArgumentParser(prog="splens")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit J-lens and R-lens transports")
    f.add_argument("model")
    f.add_argument("--out", required=True)
    f.add_argument("--rules", nargs="+", default=["j", "r"], choices=["j", "r"])
    f.add_argument("--prompts", type=int, default=25)
    f.add_argument("--length", type=int, default=128)
    f.add_argument("--stride", type=int, default=1, help="every k-th layer (4 for Ouro)")
    f.add_argument("--chunk", type=int, default=32)
    f.add_argument("--corpus-file")
    r = sub.add_parser("read", help="print SP-Lens tokens per layer for a prompt")
    r.add_argument("model")
    r.add_argument("prompt")
    r.add_argument("--lens-dir", help="fitted lenses; without it the J-free arm is used")
    r.add_argument("--position", type=int, default=-1)
    r.add_argument("--compare", nargs="*", default=[], choices=["logit", "j", "r"])
    r.add_argument("--calib-prompts", type=int, default=25)
    r.add_argument("--calib-length", type=int, default=128)
    r.add_argument("--corpus-file")
    for s in (f, r):
        s.add_argument("--device")
        s.add_argument("--dtype", default="auto")
    args = p.parse_args()

    model, tok = load_model(args.model, args.device, args.dtype)
    if args.cmd == "fit":
        fit_lenses(model, tok, args.out, rules=tuple(args.rules), prompts=args.prompts,
                   length=args.length, stride=args.stride, chunk=args.chunk,
                   corpus_file=args.corpus_file)
        print(f"saved lenses to {args.out}")
        return
    lens = SPLens(model, tok, args.lens_dir)
    lens.calibrate(args.calib_prompts, args.calib_length, args.corpus_file)
    reading = lens.read(args.prompt, args.position, tuple(args.compare))
    print(f"readout token: {reading.token!r} (position {reading.position})")
    print("\nSP-Lens")
    print(reading.table())
    for name in args.compare:
        print(f"\n{name}")
        print(reading.table(name))


if __name__ == "__main__":
    main()
