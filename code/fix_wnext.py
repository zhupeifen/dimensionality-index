# -*- coding: utf-8 -*-
"""
Repair the next-dimension coupling term and measure the cost of the repair.

As originally defined, w_next is taken from the strongest metal-metal contact
within NEXT_CUT that would raise the periodic rank, with no requirement that a
halide mediate it. In a cluster with direct metal-metal bonding the term can
therefore select a contact SHORTER than the shortest bridged separation d0,
which inverts the exponential and produces a spuriously large fractional part.
Observed in C22H18Cu4I2N8, a Cu4 cluster with cuprophilic Cu-Cu contacts at
2.51 A against a shortest bridged separation of 5.37 A.

Two guards are evaluated:

  guard A   d_next >= d0. A "next" connection closer than the nearest bridged
            pair is not a next connection. Minimal and hard to argue with.

  guard B   the contact must additionally be halide-mediated: the two metals
            must share a halide within an extended cutoff (EXT x the first-shell
            cutoff). Chemically stronger, but it will suppress the term for
            layers separated by a spacer, where nothing bridges across the gap.

Both are computed for every structure so the choice can be made on evidence,
and both are applied to the five reference copper chlorides so the cost to the
published values is visible before either is adopted.

Usage
    python fix_wnext.py <validation_multihalide.json> <cache_dir> <struct_dir> <out.json>
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SRC, CACHE, STRUCT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

LAM = 1.0
NEXT_CUT = 8.0
EXT = 1.5                      # extension factor for guard B
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


def analyse(st, metal, halides=None):
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym) if e == metal]
    pool = halides if halides else list(CUT)
    X = [(i, sym[i]) for i, e in enumerate(sym) if e in pool]
    if not M or not X or len(M) > 90:
        return None
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell, shell_ext = {}, {}
    for i in M:
        s, se = set(), set()
        for (h, hx) in X:
            d = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(d <= CUT[hx])[0]:
                s.add((h, tuple(imgs[k].astype(int))))
            for k in np.where(d <= CUT[hx] * EXT)[0]:
                se.add((h, tuple(imgs[k].astype(int))))
        shell[i], shell_ext[i] = s, se
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
        return dict(d_top=0, D_old=0.0, D_A=0.0, D_B=0.0,
                    f_old=0.0, f_A=0.0, f_B=0.0, d0=None)
    d0 = min(v[0] for v in br.values())
    w = {k: v[1] * np.exp(-(v[0] - d0) / LAM) for k, v in br.items()}
    d_top = rank(list(br))
    wmed = float(np.median(list(w.values())))

    def shares_extended(i, j, T):
        for (h, ti) in shell_ext[i]:
            for (h2, tj) in shell_ext[j]:
                if h2 == h and tuple(np.array(ti) - np.array(tj)) == T:
                    return True
        return False

    imgs2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), float)
    have = set(br)
    best = {"old": 0.0, "A": 0.0, "B": 0.0}
    for i in M:
        for j in M:
            dd = np.linalg.norm((frac[j][None, :] + imgs2 - frac[i][None, :]) @ A, axis=1)
            for k in np.where((dd > 0.1) & (dd <= NEXT_CUT))[0]:
                T = tuple(imgs2[k].astype(int))
                if (i, j, T) in have:
                    continue
                if rank(list(have) + [(i, j, T)]) <= d_top:
                    continue
                d = float(dd[k])
                cand = float(np.exp(-(d - d0) / LAM))
                best["old"] = max(best["old"], cand)
                if d >= d0:
                    best["A"] = max(best["A"], cand)
                    if shares_extended(i, j, T):
                        best["B"] = max(best["B"], cand)

    def frac_of(wn):
        return wn / (wn + wmed) if (wn + wmed) > 0 else 0.0

    fo, fa, fb = (frac_of(best["old"]), frac_of(best["A"]), frac_of(best["B"]))
    return dict(d_top=d_top, d0=round(d0, 3),
                f_old=round(fo, 4), f_A=round(fa, 4), f_B=round(fb, 4),
                D_old=round(d_top + fo, 3), D_A=round(d_top + fa, 3),
                D_B=round(d_top + fb, 3))


def main():
    from pymatgen.core import Structure
    from ase.io import read as ase_read

    # ---- cost to the published reference values -------------------------
    print("=" * 74)
    print("effect on the five reference structures in the manuscript")
    print("=" * 74)
    print(f"{'compound':<16}{'d_top':>6}{'D published':>13}{'D guard A':>11}{'D guard B':>11}")
    REF = [("Cs2CuCl4", "Cs2CuCl4"), ("Cs3Cu2Cl5", "126-0D-Cl"), ("CsCuCl3", "CsCuCl3"),
           ("CsCu2Cl3", "278-1D-Cl"), ("Cs8Cu2Sb4Cl24", "292-2D-Cl")]
    for label, src in REF:
        path = os.path.join(STRUCT, src, "CONTCAR")
        try:
            atoms = ase_read(path, format="vasp")
            st = Structure(atoms.get_cell()[:], atoms.get_chemical_symbols(),
                           atoms.get_scaled_positions())
            r = analyse(st, "Cu", ["Cl"])
        except Exception as e:
            print(f"{label:<16} could not read: {str(e)[:40]}")
            continue
        if r is None:
            print(f"{label:<16} no Cu-Cl network")
            continue
        print(f"{label:<16}{r['d_top']:>6}{r['D_old']:>13.3f}{r['D_A']:>11.3f}{r['D_B']:>11.3f}")

    # ---- effect across the validation set --------------------------------
    rows = json.load(open(SRC))
    print(f"\nre-evaluating {len(rows)} validation structures under both guards")
    out, failed = [], 0
    for n, r in enumerate(rows, 1):
        try:
            raw = open(os.path.join(CACHE, f"{r['cod']}.cif"), "rb").read()
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            res = analyse(st, r["metal"], r.get("halides_present") or None)
        except Exception:
            failed += 1
            continue
        if res is None:
            failed += 1
            continue
        q = dict(r)
        q.update(res)
        out.append(q)
        if n % 200 == 0:
            print(f"   {n}/{len(rows)}")
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n" + "=" * 74)
    print(f"re-evaluated {len(out)}  (failed {failed})")
    print("=" * 74)
    for key, name in (("f_old", "as published"), ("f_A", "guard A  d_next >= d0"),
                      ("f_B", "guard B  halide-mediated")):
        nz = [r for r in out if r[key] > 0.001]
        z = [r for r in out if r[key] <= 0.001]
        anz = sum(1 for r in nz if r["d_top"] == r["stated"])
        az = sum(1 for r in z if r["d_top"] == r["stated"])
        print(f"\n{name}:")
        print(f"   nonzero fractional part : {len(nz)} of {len(out)}")
        print(f"   agreement when nonzero  : {anz}/{len(nz)}"
              f" = {100.0*anz/max(1,len(nz)):.1f} per cent")
        print(f"   agreement when zero     : {az}/{len(z)}"
              f" = {100.0*az/max(1,len(z)):.1f} per cent")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
