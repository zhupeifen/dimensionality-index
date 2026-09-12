# -*- coding: utf-8 -*-
"""
Benchmark the sublattice-restricted index against established whole-network
dimensionality algorithms as implemented in ASE.

  RDA  Rank Determination Algorithm  (Larsen et al. scoring parameter)
  TSA  Topology-Scaling Algorithm    (Ashton et al.)

Both classify the complete bonded network. The index defined here restricts the
network to metal centres bridged by a chosen ligand. The comparison shows where
the two answers differ and why.
"""
import os, sys, json, itertools, warnings
import numpy as np

warnings.filterwarnings("ignore")

STRUCT = sys.argv[1]        # Proposals/NSF-DMR-2025/Structure
CACHE = sys.argv[2]         # COD cif cache
OUT = sys.argv[3]

LAM = 1.0
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}
NEXT_CUT = 8.0


def sublattice_index(cell, frac, sym, metals, halide, cut):
    A = np.array(cell)
    frac = np.array(frac) % 1.0
    cu = [i for i, e in enumerate(sym) if e in metals]
    xi = [i for i, e in enumerate(sym) if e == halide]
    if not cu or not xi:
        return None
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell = {}
    for i in cu:
        s = set()
        for h in xi:
            dd = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(dd <= cut)[0]:
                s.add((h, tuple(imgs[k].astype(int))))
        shell[i] = s
    br = {}
    for i in cu:
        for j in cu:
            for (h, ti) in shell[i]:
                for (h2, tj) in shell[j]:
                    if h2 != h:
                        continue
                    T = tuple(np.array(ti) - np.array(tj))
                    if i == j and T == (0, 0, 0):
                        continue
                    dd = np.linalg.norm((frac[j] + np.array(T) - frac[i]) @ A)
                    br.setdefault((i, j, T), [dd, 0])[1] += 1
    if not br:
        return dict(D=0.0, d_top=0)
    d0 = min(v[0] for v in br.values())
    w = {k: v[1] * np.exp(-(v[0] - d0) / LAM) for k, v in br.items()}

    def rank(bs):
        adj = {}
        for (i, j, T) in bs:
            adj.setdefault(i, []).append((j, np.array(T)))
        if not adj:
            return 0
        st = sorted(adj)[0]
        pos, stack, tr = {st: np.zeros(3, int)}, [st], []
        while stack:
            u = stack.pop()
            for (v, T) in adj.get(u, []):
                nv = pos[u] + T
                if v not in pos:
                    pos[v] = nv; stack.append(v)
                else:
                    dl = nv - pos[v]
                    if dl.any():
                        tr.append(dl)
        return int(np.linalg.matrix_rank(np.array(tr, float), tol=1e-6)) if tr else 0

    d_top = rank(list(br))
    wmed = float(np.median(list(w.values())))
    imgs2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), float)
    have, wnext = set(br), 0.0
    for i in cu:
        for j in cu:
            dd = np.linalg.norm((frac[j][None, :] + imgs2 - frac[i][None, :]) @ A, axis=1)
            for k in np.where((dd > 0.1) & (dd <= NEXT_CUT))[0]:
                T = tuple(imgs2[k].astype(int))
                if (i, j, T) in have:
                    continue
                if rank(list(have) + [(i, j, T)]) > d_top:
                    wnext = max(wnext, float(np.exp(-(dd[k] - d0) / LAM)))
    f = wnext / (wnext + wmed) if (wnext + wmed) > 0 else 0.0
    return dict(D=round(d_top + f, 3), d_top=d_top)


def ase_dims(atoms):
    """Whole-network dimensionality from ASE, best-scoring component."""
    from ase.geometry.dimensionality import analyze_dimensionality
    out = {}
    for method in ("RDA", "TSA"):
        try:
            res = analyze_dimensionality(atoms, method=method)
            out[method] = res[0].dimtype if res else "?"
        except Exception as e:
            out[method] = f"err"
    return out


CASES = [
    # (label, source, kind, halide)
    ("Cs2CuCl4",       "Cs2CuCl4",   "contcar", "Cl"),
    ("Cs3Cu2Cl5",      "126-0D-Cl",  "contcar", "Cl"),
    ("CsCuCl3",        "CsCuCl3",    "contcar", "Cl"),
    ("CsCu2Cl3",       "278-1D-Cl",  "contcar", "Cl"),
    ("Cs8Cu2Sb4Cl24",  "292-2D-Cl",  "contcar", "Cl"),
    ("Rb2CuCl4",       "1528214",    "cod",     "Cl"),
    ("(NH4)2CuCl4",    "1528209",    "cod",     "Cl"),
    ("CuCl (zincblende)", "9008842", "cod",     "Cl"),
    ("Cs3Cu2I5",       "1539742",    "cod",     "I"),
    ("CsCu2I3",        "7246302",    "cod",     "I"),
]


def main():
    from ase.io import read as ase_read
    rows = []
    for label, src, kind, hal in CASES:
        try:
            if kind == "contcar":
                atoms = ase_read(os.path.join(STRUCT, src, "CONTCAR"), format="vasp")
            else:
                atoms = ase_read(os.path.join(CACHE, src + ".cif"))
        except Exception as e:
            print(f"  {label}: could not read ({e})")
            continue
        cell = atoms.get_cell()[:]
        frac = atoms.get_scaled_positions()
        sym = atoms.get_chemical_symbols()
        sub = sublattice_index(cell, frac, sym, {"Cu"}, hal, CUT[hal])
        a = ase_dims(atoms)
        rows.append(dict(label=label, source=src, halide=hal,
                         whole_RDA=a["RDA"], whole_TSA=a["TSA"],
                         sub_D=None if sub is None else sub["D"],
                         sub_dtop=None if sub is None else sub["d_top"]))

    print("=" * 84)
    print("whole-network dimensionality (ASE) vs Cu-X-Cu sublattice index")
    print("=" * 84)
    print(f"{'compound':<20}{'RDA':>7}{'TSA':>7}{'sub d_top':>11}{'sub D':>9}   agree?")
    print("-" * 84)
    for r in rows:
        same = str(r["sub_dtop"]) + "D" == r["whole_RDA"]
        print(f"{r['label']:<20}{r['whole_RDA']:>7}{r['whole_TSA']:>7}"
              f"{r['sub_dtop']:>11}{r['sub_D']:>9.3f}   {'yes' if same else 'NO'}")
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    diff = [r for r in rows if str(r["sub_dtop"]) + "D" != r["whole_RDA"]]
    print(f"\ndisagreements: {len(diff)} of {len(rows)}")
    for r in diff:
        print(f"   {r['label']:<20} whole network {r['whole_RDA']}, copper sublattice {r['sub_dtop']}D")


if __name__ == "__main__":
    main()
