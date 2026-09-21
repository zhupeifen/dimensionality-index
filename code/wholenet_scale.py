# -*- coding: utf-8 -*-
"""Whole-network dimensionality against the sublattice index, at database scale.

The companion article compares the two descriptions on 644 halides assembled
around author-stated labels. That set is small and selected. Here the comparison
is run on a random sample drawn from the indexed database itself, stratified by
bridging anion class, so the divergence can be quoted for inorganic crystals
generally rather than for one curated set.

Whole-network dimensionality comes from the rank determination algorithm as
implemented in ASE. It returns a dimensionality type per structure, which may be
mixed ("02D" means zero- and two-dimensional components coexist); a mixed result
is counted as differing from any single sublattice rank, because it is.

Usage
    python wholenet_scale.py <cod_dimensionality.jsonl> <out.json>
                             [--per-class N] [--seed N] [--max-atoms N]
"""
import argparse
import collections
import json
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

CIF_ROOT = os.path.join("C:\\", "Users", "pzgft", "CODdata", "cif", "cif")


def cif_path(cod, root):
    return os.path.join(root, cod[0], cod[1:3], cod[3:5], cod + ".cif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--max-atoms", type=int, default=400)
    ap.add_argument("--cif-root", default=CIF_ROOT)
    a = ap.parse_args()

    from ase.io import read as ase_read
    from ase.geometry.dimensionality import analyze_dimensionality

    by = collections.defaultdict(list)
    for line in open(a.jsonl, encoding="utf-8"):
        r = json.loads(line)
        if r.get("D") is not None:
            by[r["anion_class"]].append(r)
    rng = random.Random(a.seed)
    sample = []
    for cls, rows in sorted(by.items()):
        take = min(a.per_class, len(rows))
        sample += rng.sample(rows, take)
        print("{:10s} population {:>7,}  sampled {:>6,}".format(cls, len(rows), take))
    rng.shuffle(sample)
    print("total sampled: {:,}\n".format(len(sample)))

    out, stats = [], collections.Counter()
    t0 = time.time()
    for n, r in enumerate(sample, 1):
        rec = {"cod": r["cod"], "cls": r["anion_class"], "d_top": r["d_top"],
               "frac": r["frac"], "nmetal": r["nmetal"], "rda": None}
        try:
            atoms = ase_read(cif_path(r["cod"], a.cif_root))
            if len(atoms) > a.max_atoms:
                stats["skipped_large"] += 1
            else:
                res = analyze_dimensionality(atoms, method="RDA")
                rec["rda"] = res[0].dimtype if res else None
                rec["natoms"] = len(atoms)
        except Exception:
            stats["read_failed"] += 1
        out.append(rec)
        if n % 500 == 0:
            el = time.time() - t0
            print("   {}/{}  {:.0f}s  eta {:.0f} min".format(
                n, len(sample), el, (len(sample) - n) * el / n / 60))

    done = [r for r in out if r["rda"]]
    same = [r for r in done if r["rda"] == "{}D".format(r["d_top"])]
    mixed = [r for r in done if len(r["rda"]) > 2]
    print("\nevaluated by the whole-network algorithm: {:,} of {:,}   {}".format(
        len(done), len(out), dict(stats)))
    print("agree with the sublattice rank : {:,}  ({:.1f}%)".format(
        len(same), 100.0 * len(same) / max(1, len(done))))
    print("differ                         : {:,}  ({:.1f}%)".format(
        len(done) - len(same), 100.0 * (len(done) - len(same)) / max(1, len(done))))
    print("of which the whole network is mixed-dimensional: {:,}  ({:.1f}% of all evaluated)".format(
        len(mixed), 100.0 * len(mixed) / max(1, len(done))))

    print("\nby anion class:")
    print("  {:10s} {:>8} {:>9} {:>9}".format("class", "n", "agree", "differ"))
    per = {}
    for cls in sorted({r["cls"] for r in done}):
        g = [r for r in done if r["cls"] == cls]
        ag = sum(1 for r in g if r["rda"] == "{}D".format(r["d_top"]))
        per[cls] = {"n": len(g), "agree": ag,
                    "pct_differ": round(100.0 * (len(g) - ag) / len(g), 1)}
        print("  {:10s} {:>8,} {:>8.1f}% {:>8.1f}%".format(
            cls, len(g), 100.0 * ag / len(g), 100.0 * (len(g) - ag) / len(g)))

    json.dump({"sampled": len(out), "evaluated": len(done),
               "agree": len(same), "mixed": len(mixed),
               "per_class": per, "seed": a.seed, "per_class_target": a.per_class,
               "rows": out},
              open(a.out, "w"), indent=1)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
