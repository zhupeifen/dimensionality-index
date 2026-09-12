# -*- coding: utf-8 -*-
"""
Two tests the manuscript needs before submission.

TEST 1  a stoichiometric baseline
    The obvious objection to a structural descriptor is that counting atoms
    already answers the question: A2MX4 is layered, AMX3 is three-dimensional,
    A4MX6 holds isolated octahedra. We therefore fit a lookup from the halide
    to metal ratio to the stated dimensionality, taking the majority label in
    each ratio bin. The lookup is fitted and scored on the SAME structures,
    which cannot be done honestly for a real model but is exactly right here:
    it is the most favourable number the baseline could possibly achieve, so
    beating it is meaningful and losing to it would be decisive.

TEST 2  whole-network algorithms at scale
    Section 5 of the manuscript compares the sublattice index with the rank
    determination and topology-scaling algorithms on ten compounds. The same
    comparison is run here over the whole validation set, so the central claim
    of the paper is stated as a rate rather than as an illustration.

Both run from cached CIF files and perform no network access.

Usage
    python baseline_and_scale.py <validation_guards.json> <cache_dir> <out.json>
"""
import collections
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SRC, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FRAME = {"Pb", "Sn", "Bi", "Sb"}
HAL = {"Cl", "Br", "I"}


def counts(formula):
    out = {}
    for sym, num in re.findall(r"([A-Z][a-z]?)([\d.]*)", formula):
        if not sym:
            continue
        out[sym] = out.get(sym, 0.0) + (float(num) if num else 1.0)
    return out


def ratio_of(r):
    c = counts(r["formula"])
    m = c.get(r["metal"], 0.0)
    x = sum(v for k, v in c.items() if k in HAL)
    return None if m <= 0 else x / m


def bin_of(v):
    """Round to the nearest half, which keeps A2MX4 and A4MX6 distinct."""
    return None if v is None else round(v * 2) / 2


def baseline(rows, name):
    binned = [(bin_of(ratio_of(r)), r["stated"], r["d_top"]) for r in rows]
    binned = [b for b in binned if b[0] is not None]
    table = {}
    for b, stated, _ in binned:
        table.setdefault(b, []).append(stated)
    rule = {b: collections.Counter(v).most_common(1)[0][0] for b, v in table.items()}
    base_ok = sum(1 for b, stated, _ in binned if rule[b] == stated)
    idx_ok = sum(1 for _, stated, dt in binned if dt == stated)
    n = len(binned)
    print(f"\n{name}  (n={n})")
    print(f"   stoichiometric lookup, fitted and scored on these same structures: "
          f"{base_ok}/{n} = {100.0*base_ok/n:.1f} per cent")
    print(f"   dimensionality index                                            : "
          f"{idx_ok}/{n} = {100.0*idx_ok/n:.1f} per cent")
    print("   the lookup it learned:")
    for b in sorted(rule):
        sup = len(table[b])
        if sup >= 5:
            share = collections.Counter(table[b]).most_common(1)[0][1] / sup
            print(f"      X/M = {b:<5} -> {rule[b]}D   (n={sup}, purity {share:.0%})")
    return dict(n=n, baseline=base_ok, index=idx_ok, rule={str(k): v for k, v in rule.items()})


def main():
    rows = json.load(open(SRC))
    print("=" * 78)
    print("TEST 1  can counting atoms do the same job?")
    print("=" * 78)
    b_all = baseline(rows, "all 703 structures")
    b_frame = baseline([r for r in rows if r["metal"] in FRAME],
                       "444 Pb/Sn/Bi/Sb halide frameworks")

    print("\n" + "=" * 78)
    print("TEST 2  whole-network algorithms across the validation set")
    print("=" * 78)
    from ase.io import read as ase_read
    from ase.geometry.dimensionality import analyze_dimensionality

    out, t0 = [], time.time()
    stats = collections.Counter()
    for n, r in enumerate(rows, 1):
        path = os.path.join(CACHE, f"{r['cod']}.cif")
        rec = dict(cod=r["cod"], metal=r["metal"], stated=r["stated"],
                   d_top=r["d_top"], D=r.get("D_A", r.get("D")), rda=None, tsa=None)
        try:
            atoms = ase_read(path)
            if len(atoms) > 400:
                stats["skipped_large"] += 1
                out.append(rec)
                continue
            for meth, key in (("RDA", "rda"), ("TSA", "tsa")):
                try:
                    res = analyze_dimensionality(atoms, method=meth)
                    rec[key] = res[0].dimtype if res else None
                except Exception:
                    rec[key] = None
        except Exception:
            stats["read_failed"] += 1
        out.append(rec)
        if n % 100 == 0:
            print(f"   {n}/{len(rows)}  ({time.time()-t0:.0f}s)")

    done = [r for r in out if r["rda"]]
    same = [r for r in done if r["rda"] == f"{r['d_top']}D"]
    print(f"\n   evaluated by RDA: {len(done)} of {len(rows)}   {dict(stats)}")
    print(f"   whole network and copper/metal sublattice agree: "
          f"{len(same)}/{len(done)} = {100.0*len(same)/max(1,len(done)):.1f} per cent")
    print(f"   they DIFFER for {len(done)-len(same)} structures "
          f"({100.0*(len(done)-len(same))/max(1,len(done)):.1f} per cent)")

    print("\n   whole-network answers, by what the authors stated:")
    for s_ in (0, 1, 2, 3):
        g = [r for r in done if r["stated"] == s_]
        if not g:
            continue
        rda_ok = sum(1 for r in g if r["rda"] == f"{s_}D")
        idx_ok = sum(1 for r in g if r["d_top"] == s_)
        print(f"      stated {s_}D (n={len(g):<4}) RDA correct {rda_ok:<4} "
              f"index correct {idx_ok}")

    g_all = [r for r in done]
    rda_ok = sum(1 for r in g_all if r["rda"] == f"{r['stated']}D")
    idx_ok = sum(1 for r in g_all if r["d_top"] == r["stated"])
    print(f"\n   overall against the authors' labels, on the same {len(g_all)} structures:")
    print(f"      RDA (whole network)  : {rda_ok}/{len(g_all)} = "
          f"{100.0*rda_ok/max(1,len(g_all)):.1f} per cent")
    print(f"      sublattice index     : {idx_ok}/{len(g_all)} = "
          f"{100.0*idx_ok/max(1,len(g_all)):.1f} per cent")

    json.dump(dict(baseline_all=b_all, baseline_frame=b_frame, rows=out),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
