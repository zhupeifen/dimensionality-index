# -*- coding: utf-8 -*-
"""
Re-index the whole validation set with the framework metals taken as one
sublattice, which is the convention Section 3 of the manuscript adopts.

Two conventions are stated there. The halides are taken as a single bridging
class, which reindex_multihalide.py already applies to all 703 structures. The
framework metals are likewise taken together, because a mixed-metal halide
evaluated on one metal reports a fragment of its own network - but that second
convention had only ever been applied to the 444 Pb/Sn/Bi/Sb halides, in
validation_allmetal.json. The remaining 259 structures were still indexed on a
single named metal, so any figure quoted "on all 703" mixed the two conventions.
This script removes that inconsistency by applying both conventions throughout.

The metal sublattice is every site that is not a halide and not one of the
light elements that make up an organic cation. Structures containing a
competing inorganic network former were already excluded when the validation
set was assembled, so nothing else needs excluding here.

Reads CIFs from a local copy of the COD bulk archive, or from a flat cache
directory of <cod>.cif files; performs no network access.

Usage
    python reindex_allmetal.py <validation_guards.json> <out.json> [cif_root]
"""
import itertools
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

LAM = 1.0
NEXT_CUT = 8.0
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}
# What is not a framework metal: the bridging halides themselves, the light
# elements an organic cation is built from, and the A-site cations.
#
# The alkali metals and thallium have to be excluded explicitly. They are metals,
# but they are spacers here, not network formers - Section 4 of the manuscript
# defines the A site as "an alkali cation or thallium". Counting them into the
# sublattice links every structure that contains one in all three directions,
# which is precisely the whole-network answer the index exists to avoid. It also
# pushes large cells past the 90-centre limit: Rb23Sb9Cl54 presents 128 centres
# with rubidium counted and 36 without.
NONMETAL = {"H", "D", "C", "N", "O", "S", "Se", "Te", "P", "F", "As", "B", "Si"}
# Alkaline earths are listed for the same reason, and to state the same
# convention as reindex_chalc_multimetal.py, whose SPECTATOR set excludes them.
# Neither they nor thallium occur anywhere in the 703 halides, so the two
# definitions coincide on this data; they are written out so that the two
# scripts cannot drift if either set is extended.
ASITE = {"Li", "Na", "K", "Rb", "Cs", "Fr",
         "Be", "Mg", "Ca", "Sr", "Ba", "Tl"}

SRC = sys.argv[1]
OUT = sys.argv[2]
ROOT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    "C:\\", "Users", "pzgft", "CODdata", "cif", "cif")


def cif_path(cod):
    nested = os.path.join(ROOT, cod[0], cod[1:3], cod[3:5], cod + ".cif")
    if os.path.exists(nested):
        return nested
    flat = os.path.join(ROOT, cod + ".cif")
    return flat if os.path.exists(flat) else None


def symbols(st):
    out = []
    for s in st.sites:
        try:
            out.append(s.specie.symbol)
        except AttributeError:                      # disordered site
            out.append(max(s.species, key=s.species.get).symbol)
    return out


def rank(bonds):
    """Periodic rank of the bridging network, from the translations the walk closes on."""
    adj = {}
    for (i, j, T) in bonds:
        adj.setdefault(i, []).append((j, np.array(T, int)))
        adj.setdefault(j, []).append((i, -np.array(T, int)))
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


def index_allmetal(st):
    """Index the whole metal sublattice, bridged by any halide present."""
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym)
         if e not in NONMETAL and e not in CUT and e not in ASITE]
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
        return dict(D=0.0, d_top=0, frac=0.0, nbridge=0, d_next=None, d0=None, nmetal=len(M))
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
                d_next=None if dnext is None else round(dnext, 3), d0=round(d0, 3),
                nmetal=len(M))


def main():
    from pymatgen.core import Structure
    rows = json.load(open(SRC))
    if isinstance(rows, dict):
        rows = rows.get("rows", list(rows.values()))
    print("re-indexing %d structures on the full metal sublattice" % len(rows))

    out, failed, missing = [], 0, 0
    for n, r in enumerate(rows, 1):
        p = cif_path(r["cod"])
        if p is None:
            missing += 1
            continue
        try:
            raw = open(p, "rb").read()
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            res = index_allmetal(st)
        except Exception:
            failed += 1
            continue
        if res is None:
            failed += 1
            continue
        q = dict(r)
        q.update(res)
        out.append(q)
        if n % 100 == 0:
            print("   %d/%d" % (n, len(rows)))

    json.dump(out, open(OUT, "w"), indent=1)

    FRAME = {"Pb", "Sn", "Bi", "Sb"}

    def report(rs, title):
        if not rs:
            return
        a = sum(1 for r in rs if r["d_top"] == r["stated"])
        print("\n%s: %d/%d = %.1f per cent" % (title, a, len(rs), 100.0 * a / len(rs)))

    print("\n" + "=" * 74)
    print("re-indexed %d   failed %d   CIF not held locally %d" % (len(out), failed, missing))
    print("=" * 74)
    report(out, "ALL, framework metals as one sublattice")
    report([r for r in out if r["metal"] in FRAME], "  Pb/Sn/Bi/Sb halides")
    report([r for r in out if r["metal"] not in FRAME], "  Cu/Ag/Cd/Mn halides")


if __name__ == "__main__":
    main()
