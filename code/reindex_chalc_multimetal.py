# -*- coding: utf-8 -*-
"""
Re-index the chalcogenides treating the framework metals as one sublattice.

The first pass evaluated each structure on a single metal, which is wrong for
the mixed-metal chalcogenides that dominate the labelled set: in Ta2PdS6 the
slab is built from Ta and Pd together, and indexing on Ta alone finds a chain
where the authors, correctly, see a layer. Thirty-nine of ninety-one
disagreements contained no bridge at all.

This is the same error as evaluating a mixed-halide perovskite on one halide.
The bridging ligand is the chalcogen set; the bridged network is the set of
framework metals, meaning every metal that is not an alkali or alkaline-earth
counter-cation.

Runs from the cached CIFs; no network access.

Usage
    python reindex_chalc_multimetal.py <validation_chalcogenides.json> <cache> <out.json>
"""
import itertools
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SRC, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LAM, NEXT_CUT = 1.0, 8.0
CUT = {"S": 3.00, "Se": 3.15, "Te": 3.40}
# counter-cations: present for charge balance, not part of the framework
SPECTATOR = {"Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba",
             "H", "C", "N", "O", "F", "Cl", "Br", "I", "P"}


def symbols(st):
    out = []
    for s in st.sites:
        try:
            out.append(s.specie.symbol)
        except Exception:
            try:
                out.append(max(s.species, key=s.species.get).symbol)
            except Exception:
                out.append(None)
    return out


def rank(bonds):
    adj = {}
    for (i, j, T) in bonds:
        adj.setdefault(i, []).append((j, np.array(T)))
    if not adj:
        return 0
    start = sorted(adj)[0]
    pos, stack, trans = {start: np.zeros(3, int)}, [start], []
    while stack:
        u = stack.pop()
        for (v, T) in adj.get(u, []):
            nv = pos[u] + T
            if v not in pos:
                pos[v] = nv
                stack.append(v)
            else:
                dl = nv - pos[v]
                if dl.any():
                    trans.append(dl)
    return int(np.linalg.matrix_rank(np.array(trans, float), tol=1e-6)) if trans else 0


def index_multi(st):
    """Index every framework metal bridged by any chalcogen present."""
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym) if e and e not in SPECTATOR and e not in CUT]
    X = [(i, sym[i]) for i, e in enumerate(sym) if e in CUT]
    if not M or not X or len(M) > 90:
        return None
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell = {}
    for i in M:
        s = set()
        for (h, hx) in X:
            d = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(d <= CUT[hx])[0]:
                s.add((h, tuple(imgs[k].astype(int))))
        shell[i] = s
    br = {}
    for i in M:
        for j in M:
            for (h, ti) in shell[i]:
                for (h2, tj) in shell[j]:
                    if h2 != h:
                        continue
                    T = tuple(np.array(ti) - np.array(tj))
                    if i == j and T == (0, 0, 0):
                        continue
                    d = np.linalg.norm((frac[j] + np.array(T) - frac[i]) @ A)
                    br.setdefault((i, j, T), [d, 0])[1] += 1
    if not br:
        return dict(D=0.0, d_top=0, frac=0.0, nbridge=0, nmetal=len(M))
    d0 = min(v[0] for v in br.values())
    w = {k: v[1] * np.exp(-(v[0] - d0) / LAM) for k, v in br.items()}
    d_top = rank(list(br))
    wmed = float(np.median(list(w.values())))
    imgs2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), float)
    have, wn = set(br), 0.0
    for i in M:
        for j in M:
            dd = np.linalg.norm((frac[j][None, :] + imgs2 - frac[i][None, :]) @ A, axis=1)
            for k in np.where((dd >= d0) & (dd <= NEXT_CUT))[0]:
                T = tuple(imgs2[k].astype(int))
                if (i, j, T) in have:
                    continue
                if rank(list(have) + [(i, j, T)]) > d_top:
                    wn = max(wn, float(np.exp(-(dd[k] - d0) / LAM)))
    f = wn / (wn + wmed) if (wn + wmed) > 0 else 0.0
    return dict(D=round(d_top + f, 3), d_top=d_top, frac=round(f, 4),
                nbridge=len(br), nmetal=len(M))


def main():
    from pymatgen.core import Structure
    rows = json.load(open(SRC))
    out, fail = [], 0
    print(f"re-indexing {len(rows)} chalcogenides on the full framework-metal sublattice")
    for n, r in enumerate(rows, 1):
        try:
            raw = open(os.path.join(CACHE, f"{r['cod']}.cif"), "rb").read()
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            res = index_multi(st)
        except Exception:
            fail += 1
            continue
        if res is None:
            fail += 1
            continue
        q = dict(r)
        q["d_top_single"] = r["d_top"]
        q.update(res)
        out.append(q)
        if n % 50 == 0:
            print(f"   {n}/{len(rows)}")
    json.dump(out, open(OUT, "w"), indent=1)

    a_new = sum(1 for r in out if r["d_top"] == r["stated"])
    a_old = sum(1 for r in out if r["d_top_single"] == r["stated"])
    print("\n" + "=" * 72)
    print(f"re-indexed {len(out)}  (failed {fail})")
    print("=" * 72)
    print(f"  single metal, as first run : {a_old}/{len(out)} = {100.0*a_old/len(out):.1f} per cent")
    print(f"  full framework sublattice  : {a_new}/{len(out)} = {100.0*a_new/len(out):.1f} per cent")
    print("\n  confusion (rows authors, cols index):")
    print("        " + "".join(f"{c:>7}" for c in (0, 1, 2, 3)))
    for s_ in (0, 1, 2, 3):
        nn = sum(1 for r in out if r["stated"] == s_)
        if not nn:
            continue
        print(f"   {s_}D  " + "".join(
            f"{sum(1 for r in out if r['stated']==s_ and r['d_top']==c):>7}" for c in (0, 1, 2, 3))
            + f"   (n={nn})")

    both = [r for r in out if r.get("rda")]
    if both:
        same = sum(1 for r in both if r["rda"] == f"{r['d_top']}D")
        rda_ok = sum(1 for r in both if r["rda"] == f"{r['stated']}D")
        idx_ok = sum(1 for r in both if r["d_top"] == r["stated"])
        print(f"\n  THE KEY COMPARISON, {len(both)} structures both methods evaluated:")
        print(f"     sublattice and whole network AGREE : {same}/{len(both)}"
              f" = {100.0*same/len(both):.1f} per cent")
        print(f"     whole network vs authors' label    : {rda_ok}/{len(both)}"
              f" = {100.0*rda_ok/len(both):.1f} per cent")
        print(f"     sublattice   vs authors' label     : {idx_ok}/{len(both)}"
              f" = {100.0*idx_ok/len(both):.1f} per cent")
        print("\n     halides for comparison: methods agreed 20.1 per cent,"
              "\n     whole network matched labels 17.0 per cent, sublattice 71.4 per cent")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
