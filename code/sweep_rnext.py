# -*- coding: utf-8 -*-
"""Sensitivity of the index to the candidate search radius R_next.

R_next bounds the search for a non-bridged metal pair whose addition would
raise the periodic rank. The rank itself is computed from the bridging network
before any candidate is considered, so R_next cannot affect it; this script
tests that rather than assuming it, and measures what the fractional term does
across the radius.

Structures with no metal-ligand-metal bridge have no median coupling to compare
against, so the fractional term is undefined for them. They are held out of the
zero-versus-nonzero comparison here, as in Section 3 of the manuscript, instead
of being counted as measured zeros.

Usage
    python sweep_rnext.py <validation_guards.json> <out.json> [cif_root]

Writes one record per structure per radius, so the table in Section 3 can be
regenerated without re-running the indexer.
"""
import importlib.util
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

RADII = (4.0, 6.0, 7.0, 7.5, 8.0, 10.0, 12.0)
HERE = os.path.dirname(os.path.abspath(__file__))

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "validation_guards.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "sweep_rnext.json")
ROOT = sys.argv[3] if len(sys.argv) > 3 else None


def load_indexer():
    """Import reindex_allmetal without running its command-line main()."""
    path = os.path.join(HERE, "reindex_allmetal.py")
    saved = sys.argv
    sys.argv = ["reindex_allmetal.py", SRC, os.devnull] + ([ROOT] if ROOT else [])
    spec = importlib.util.spec_from_file_location("ram", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


def main():
    from pymatgen.core import Structure
    ram = load_indexer()
    rows = json.load(open(SRC))
    if isinstance(rows, dict):
        rows = rows.get("rows", list(rows.values()))

    # Parse each CIF once; the sweep then costs only the indexing.
    parsed = []
    for r in rows:
        p = ram.cif_path(r["cod"])
        if p is None:
            continue
        try:
            st = Structure.from_str(open(p, "rb").read().decode("utf-8", "replace"),
                                    fmt="cif")
        except Exception:
            continue
        parsed.append((r, st))
    print("parsed %d of %d structures" % (len(parsed), len(rows)))

    out = {}
    base_rank = {}
    for R in RADII:
        ram.NEXT_CUT = R
        recs = []
        for r, st in parsed:
            try:
                res = ram.index_allmetal(st)
            except Exception:
                continue
            if res is None:
                continue
            recs.append(dict(cod=r["cod"], stated=r["stated"], d_top=res["d_top"],
                             frac=res["frac"], nbridge=res["nbridge"],
                             d_next=res["d_next"]))
        out["%.1f" % R] = recs

        defined = [q for q in recs if q["nbridge"] > 0]
        nz = [q for q in defined if q["frac"] != 0]
        z = [q for q in defined if q["frac"] == 0]
        ok = lambda g: sum(1 for q in g if q["d_top"] == q["stated"])
        if not base_rank:
            base_rank = {q["cod"]: q["d_top"] for q in recs}
            moved = 0
        else:
            moved = sum(1 for q in recs if base_rank.get(q["cod"]) != q["d_top"])
        print("R = %5.1f A   indexed %d   rank changes %d   nonzero f %3d   "
              "agree f=0 %3d/%-3d = %4.1f%%   agree f>0 %2d/%-2d = %4.1f%%   "
              "overall %d/%d = %.1f%%"
              % (R, len(recs), moved, len(nz),
                 ok(z), len(z), 100.0 * ok(z) / max(len(z), 1),
                 ok(nz), len(nz), 100.0 * ok(nz) / max(len(nz), 1),
                 ok(recs), len(recs), 100.0 * ok(recs) / len(recs)))

    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote " + OUT)


if __name__ == "__main__":
    main()
