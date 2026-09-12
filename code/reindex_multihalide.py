# -*- coding: utf-8 -*-
"""
Re-index the validation set treating the halides as a single bridging class.

The first pass evaluated each structure on one metal-halide pair. That is wrong
for mixed-halide compounds, which are common: a (PbBr(x)I(1-x)) sheet indexed on
Pb-Cl alone contains no bridges and is reported as zero-dimensional. The bridging
ligand is the halide *set*, not one element of it.

Runs entirely from the cached CIF files written by harvest_validation.py, so it
performs no network access.

Usage
    python reindex_multihalide.py <validation_set.json> <cache_dir> <out.json>
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

LAM = 1.0
NEXT_CUT = 8.0
# per-halide first-shell cutoffs; a bridge may use any halide
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}
COMPETING = {"O", "S", "Se", "Te", "P", "F", "As", "B", "Si"}
FALSEPOS = re.compile(
    r"3D\s*(electron\s*)?(diffraction|ED\b|printing|structure\s*determination)|"
    r"\b3d\b[\s-]*(complex|transition|metal|ion|orbital|block|4f)|\b[12]D\s*NMR\b", re.I)


def elements(formula):
    return set(re.findall(r"([A-Z][a-z]?)", formula))


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


def index_multi(st, metal):
    """Index the metal sublattice bridged by ANY halide present."""
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym) if e == metal]
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
        return dict(D=0.0, d_top=0, frac=0.0, nbridge=0, d_next=None, d0=None)
    d0 = min(v[0] for v in br.values())
    w = {k: v[1] * np.exp(-(v[0] - d0) / LAM) for k, v in br.items()}
    d_top = rank(list(br))
    wmed = float(np.median(list(w.values())))
    imgs2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), float)
    have, wnext, dnext = set(br), 0.0, None
    for i in M:
        for j in M:
            dd = np.linalg.norm((frac[j][None, :] + imgs2 - frac[i][None, :]) @ A, axis=1)
            for k in np.where((dd > 0.1) & (dd <= NEXT_CUT))[0]:
                T = tuple(imgs2[k].astype(int))
                if (i, j, T) in have:
                    continue
                if rank(list(have) + [(i, j, T)]) > d_top:
                    c = float(np.exp(-(dd[k] - d0) / LAM))
                    if c > wnext:
                        wnext, dnext = c, float(dd[k])
    f = wnext / (wnext + wmed) if (wnext + wmed) > 0 else 0.0
    return dict(D=round(d_top + f, 3), d_top=d_top, frac=round(f, 4), nbridge=len(br),
                d_next=None if dnext is None else round(dnext, 3), d0=round(d0, 3))


def main():
    from pymatgen.core import Structure
    src = json.load(open(SRC))["rows"]
    keep = [r for r in src
            if not (elements(r["formula"]) & COMPETING) and not FALSEPOS.search(r["text"])]
    print(f"re-indexing {len(keep)} structures (halide-only anion, no extractor false positives)")

    out, failed = [], 0
    for n, r in enumerate(keep, 1):
        try:
            raw = open(os.path.join(CACHE, f"{r['cod']}.cif"), "rb").read()
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            res = index_multi(st, r["metal"])
        except Exception:
            failed += 1
            continue
        if res is None:
            failed += 1
            continue
        q = dict(r)
        q["halides_present"] = sorted(elements(r["formula"]) & set(CUT))
        q.update(res)
        out.append(q)
        if n % 150 == 0:
            print(f"   {n}/{len(keep)}")

    json.dump(out, open(OUT, "w"), indent=1)
    FRAME = {"Pb", "Sn", "Bi", "Sb"}

    def report(rows, title):
        if not rows:
            return
        a = sum(1 for r in rows if r["d_top"] == r["stated"])
        print(f"\n{title}: {a}/{len(rows)} = {100.0*a/len(rows):.1f} per cent")
        print("        " + "".join(f"{c:>7}" for c in (0, 1, 2, 3)))
        for s_ in (0, 1, 2, 3):
            n_ = sum(1 for r in rows if r["stated"] == s_)
            if not n_:
                continue
            print(f"  {s_}D  " + "".join(
                f"{sum(1 for r in rows if r['stated']==s_ and r['d_top']==c):>7}"
                for c in (0, 1, 2, 3)) + f"   (n={n_})")

    print("\n" + "=" * 74)
    print(f"re-indexed {len(out)}   failed {failed}")
    print("=" * 74)
    report([r for r in out if r["metal"] in FRAME],
           "PRIMARY VALIDATION  Pb/Sn/Bi/Sb halide frameworks")
    report([r for r in out if r["metal"] not in FRAME],
           "COORDINATION CHEMISTRY  Cu/Ag/Cd/Mn (halide competes with other bridges)")
    mixed = [r for r in out if len(r["halides_present"]) > 1]
    a = sum(1 for r in mixed if r["d_top"] == r["stated"])
    print(f"\nmixed-halide subset: {a}/{len(mixed)} = "
          f"{100.0*a/max(1,len(mixed)):.1f} per cent  (the first pass got these wrong by construction)")
    nz = [r for r in out if r["frac"] > 0.001]
    print(f"fractional part nonzero: {len(nz)} of {len(out)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
