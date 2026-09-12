# -*- coding: utf-8 -*-
"""
Measure what the index actually costs to evaluate, and where it stops working.

The manuscript says the index "evaluates in seconds", which is not a number a
reader can plan a screen around. This times the evaluation over the cached
validation set and reports throughput against structure size, together with the
size at which the O(N^2) coordination-shell search becomes the limit.

Runs from cached CIF files; no network access.

Usage
    python benchmark_speed.py <validation_guards.json> <cache_dir> <out.json>
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SRC, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
LAM, NEXT_CUT = 1.0, 8.0
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}


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


def index_of(st, metal, halides):
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym) if e == metal]
    X = [(i, sym[i]) for i, e in enumerate(sym) if e in (halides or CUT)]
    if not M or not X:
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
        return dict(D=0.0, d_top=0)
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
    return dict(D=round(d_top + f, 3), d_top=d_top)


def main():
    from pymatgen.core import Structure
    rows = json.load(open(SRC))
    out = []
    print(f"timing {len(rows)} structures")
    for n, r in enumerate(rows, 1):
        p = os.path.join(CACHE, f"{r['cod']}.cif")
        try:
            raw = open(p, "rb").read()
            t0 = time.perf_counter()
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            t_parse = time.perf_counter() - t0
            nm = sum(1 for s in st.sites
                     if getattr(s.specie, "symbol", None) == r["metal"])
            t1 = time.perf_counter()
            res = index_of(st, r["metal"], r.get("halides_present"))
            t_index = time.perf_counter() - t1
        except Exception:
            continue
        if res is None:
            continue
        out.append(dict(cod=r["cod"], nsites=len(st), nmetal=nm,
                        t_parse=round(t_parse, 5), t_index=round(t_index, 5)))
        if n % 150 == 0:
            print(f"   {n}/{len(rows)}")

    json.dump(out, open(OUT, "w"), indent=1)
    ti = np.array([r["t_index"] for r in out])
    ns = np.array([r["nsites"] for r in out])
    nm = np.array([r["nmetal"] for r in out])
    print("\n" + "=" * 68)
    print(f"timed {len(out)} structures")
    print("=" * 68)
    print(f"  index evaluation, median      : {np.median(ti)*1000:.1f} ms")
    print(f"  index evaluation, 90th pct    : {np.percentile(ti,90)*1000:.1f} ms")
    print(f"  index evaluation, slowest     : {ti.max()*1000:.0f} ms "
          f"({ns[ti.argmax()]} sites, {nm[ti.argmax()]} metal centres)")
    print(f"  total for the whole set       : {ti.sum():.1f} s")
    print(f"  throughput                    : {len(out)/ti.sum():.0f} structures/s")
    print(f"\n  structure size, median        : {int(np.median(ns))} sites, "
          f"{int(np.median(nm))} metal centres")
    print(f"  largest handled               : {ns.max()} sites, {nm.max()} metal centres")
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 200)):
        s = (nm >= lo) & (nm < hi)
        if s.sum():
            print(f"  {lo:>3}-{hi:<4} metal centres (n={s.sum():>3}): "
                  f"median {np.median(ti[s])*1000:7.1f} ms")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
