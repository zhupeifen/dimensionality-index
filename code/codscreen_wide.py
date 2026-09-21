# -*- coding: utf-8 -*-
"""Copper-halide screen, rebuilt from the COD-wide dimensionality index.

Oxidation state comes from charge balance on A(x)Cu(y)X(z) with A an alkali
cation or thallium: n(Cu) = (z - x)/y, so 1 is Cu(I) and 2 is Cu(II). Anything
that does not reduce to an integer, or that carries another element, is dropped
rather than guessed at.

One correction matters. The database-wide index treats every non-spectator metal
as a single sublattice, which is right in general and wrong for this question:
thallium is a framework metal there, so a Tl-Cu iodide is scored on the combined
Tl+Cu network. Tl2CuI3 indexes 3.000 that way and 0.005 on copper alone - the
same failure the antimony compound shows. Every candidate is therefore
re-indexed here with Tl demoted to a counter-cation, so the number reported is
the dimensionality of the copper-halide network and nothing else.

Usage
    python codscreen_wide.py <cod_dimensionality.jsonl> [-o out.json]
"""
import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

A_SITE = {"Li", "Na", "K", "Rb", "Cs", "Tl"}
HALIDE = {"Cl", "Br", "I"}
ELEM = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
OURS = {"Cs2CuCl4": 0.000, "Cs3Cu2Cl5": 0.000, "CsCuCl3": 1.005, "CsCu2Cl3": 1.026}

DEF_CIF = os.path.join("C:\\", "Users", "pzgft", "CODdata", "cif", "cif")
DEF_IDX = os.path.join("C:\\", "Users", "pzgft", "OneDrive - University of Missouri",
                       "Manuscript", "Dimensionality Index", "cod_wide_index.py")


def parse(formula):
    out = {}
    for sym, cnt in ELEM.findall(formula):
        if sym:
            out[sym] = out.get(sym, 0.0) + (float(cnt) if cnt else 1.0)
    return out


def classify(formula):
    c = parse(formula)
    if "Cu" not in c:
        return None
    hal = [e for e in c if e in HALIDE]
    if len(hal) != 1:
        return None
    z, y = c[hal[0]], c["Cu"]
    others = set(c) - {"Cu", hal[0]}
    if not others:
        n = z / y
        if abs(n - round(n)) > 1e-6:
            return None
        return "binary1" if abs(n - 1) < 1e-6 else "binary2" if abs(n - 2) < 1e-6 else "binary"
    if not others <= A_SITE:
        return None
    x = sum(c[e] for e in others)
    n = (z - x) / y
    return 1 if abs(n - 1) < 1e-6 else 2 if abs(n - 2) < 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--out", default="screen_summary_wide.json")
    ap.add_argument("--cif-root", default=DEF_CIF)
    ap.add_argument("--indexer", default=DEF_IDX)
    a = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("cwi", a.indexer)
    cwi = importlib.util.module_from_spec(spec)
    keep, sys.argv = sys.argv, ["x", "a", "b"]
    spec.loader.exec_module(cwi)
    sys.argv = keep
    cwi.SPECTATOR = set(cwi.SPECTATOR) | {"Tl"}
    from pymatgen.core import Structure

    def cif(cod):
        return os.path.join(a.cif_root, cod[0], cod[1:3], cod[3:5], cod + ".cif")

    cu1, cu2, binary = [], [], []
    bin1, bin2, binX = [], [], []
    changed, failed = [], 0
    for line in open(a.jsonl, encoding="utf-8"):
        r = json.loads(line)
        if r.get("D") is None or r.get("anion_class") != "halide":
            continue
        k = classify(r["formula"])
        if k is None:
            continue
        D, top = r["D"], r["d_top"]
        try:
            res, _ = cwi.index_structure(Structure.from_file(cif(r["cod"])))
            if res:
                if res["d_top"] != top:
                    changed.append((r["cod"], r["formula"], top, res["d_top"]))
                D, top = res["D"], res["d_top"]
            else:
                failed += 1
        except Exception:
            failed += 1
        if k == 1:
            cu1.append((r["formula"], D, top, r["cod"]))
        elif k == 2:
            cu2.append((r["formula"], D, top, r["cod"]))
        else:
            binary.append((r["formula"], D, top, r["cod"]))
            (bin1 if k == "binary1" else bin2 if k == "binary2" else binX).append(
                (r["formula"], D, top, r["cod"]))

    def counts(rows):
        c = Counter(t for _f, _D, t, _c in rows)
        return [c.get(i, 0) for i in range(4)]

    S = {"Cu1": counts(cu1), "Cu1_n": len(cu1),
         "Cu2": counts(cu2), "Cu2_n": len(cu2),
         "Cu1_D": sorted(D for _f, D, _t, _c in cu1),
         "Cu2_D": sorted(D for _f, D, _t, _c in cu2),
         "binary_all": counts(binary), "binary_all_n": len(binary),
         "binary_Cu1": counts(bin1), "binary_Cu1_n": len(bin1),
         "binary_Cu2": counts(bin2), "binary_Cu2_n": len(bin2),
         "ours": OURS}
    json.dump(S, open(a.out, "w"), indent=1)

    print(f"re-indexed on the copper sublattice: {len(changed)} changed dimension, {failed} unreadable")
    for cod, f, was, now in changed:
        print(f"    COD {cod:>8}  {f:<14} {was}D -> {now}D  (second metal was carrying the network)")
    print()
    print(f"{'':10s}{'0D':>6}{'1D':>6}{'2D':>6}{'3D':>6}   n")
    for name, rows in (("Cu(I)", cu1), ("Cu(II)", cu2), ("binary", binary)):
        print(f"{name:10s}" + "".join(f"{v:>6}" for v in counts(rows)) + f"   {len(rows)}")
    hi = [r for r in cu1 if r[2] >= 2]
    print(f"\nCu(I) reaching 2D or above on the copper network: {len(hi)}"
          + ("   -> the bound holds" if not hi else ""))
    for f, D, t, c in hi:
        print(f"    COD {c}  {f}  D={D}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
