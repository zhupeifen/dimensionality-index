# -*- coding: utf-8 -*-
"""
Test the index outside the halides, on the family the prior methods were built for.

The validation so far is halides only, which leaves the index looking like a
tool for one chemistry. It also leaves a sharper question unasked: the whole-
network algorithms were developed for van der Waals layered materials, where the
metal-chalcogenide slab IS the bonded component. There the sublattice
restriction should change nothing, and the index should agree both with those
algorithms and with the depositing authors.

If it does, the paper's claim becomes symmetric and much stronger: the index
agrees where agreement is correct, and diverges only where the whole network
carries connectivity the sublattice does not. Divergence would then be a
property of the material class, not of the method.

Same procedure as harvest_validation.py: COD metadata is filtered for entries
whose depositing authors state a dimensionality, only those are downloaded, and
the index is evaluated on the metal-chalcogen sublattice.

Usage
    python harvest_chalcogenides.py <out.json> [cache_dir]
"""
import itertools
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OUT = sys.argv[1]
CACHE = sys.argv[2] if len(sys.argv) > 2 else "codcache_chalc"
os.makedirs(CACHE, exist_ok=True)

LAM = 1.0
NEXT_CUT = 8.0
# first-shell cutoffs per chalcogen, from typical metal-chalcogen bond lengths
CUT = {"S": 3.00, "Se": 3.15, "Te": 3.40}
# transition and post-transition metals that form layered chalcogenides
METALS = ["Mo", "W", "Ti", "Nb", "Ta", "V", "Sn", "Bi", "Sb", "In",
          "Ga", "Ge", "Zr", "Hf", "Re", "Pt", "Pd", "Fe", "Cr"]
# a competing anion would mean the chalcogen is not the bridging ligand
COMPETING = {"O", "F", "Cl", "Br", "I", "N", "P"}

LABELS = [
    (3, r"\bthree[- ]dimensional\b|\b3D\b(?!\s*(?:material|electron|printing))"),
    (2, r"\btwo[- ]dimensional\b|\b2D\b(?!\s*material)|\blayered\b|\bmonolayer\b|\bvan der Waals\b"),
    (1, r"\bone[- ]dimensional\b|\b1D\b|\bchain[- ]like\b|\bribbon\b"),
    (0, r"\bzero[- ]dimensional\b|\b0D\b|\bmolecular cluster\b"),
]
VETO = re.compile(r"quasi[- ]|from\s+\dD\s+to\s+\dD|dimensional\s+reduction|"
                  r"3D\s*(electron\s*)?diffraction|\b3d\b[\s-]*(complex|transition|metal|orbital)", re.I)


def fetch(url, path, pause=0.5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "dimindex-chalc/1.0"})
    d = urllib.request.urlopen(req, timeout=90).read()
    open(path, "wb").write(d)
    time.sleep(pause)
    return d


def cod(**kw):
    url = "https://www.crystallography.net/cod/result?" + urllib.parse.urlencode(dict(kw, format="json"))
    key = "q_" + re.sub(r"\W+", "_", str(sorted(kw.items())))[:90] + ".json"
    try:
        return json.loads(fetch(url, os.path.join(CACHE, key), pause=1.0).decode("utf-8", "replace"))
    except Exception as e:
        print(f"    query failed {kw}: {e}")
        return []


def label_of(rec):
    for field in ("title", "chemname"):
        t = (rec.get(field) or "").strip()
        if not t or VETO.search(t):
            continue
        for dim, pat in LABELS:
            m = re.search(pat, t, re.I)
            if m:
                return dim, m.group(0), field
    return None


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


def index_on(st, metal, chalcs):
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = symbols(st)
    M = [i for i, e in enumerate(sym) if e == metal]
    X = [(i, sym[i]) for i, e in enumerate(sym) if e in chalcs]
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
        return dict(D=0.0, d_top=0, frac=0.0, nbridge=0)
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
    return dict(D=round(d_top + f, 3), d_top=d_top, frac=round(f, 4), nbridge=len(br))


def main():
    from pymatgen.core import Structure
    from ase.io import read as ase_read
    from ase.geometry.dimensionality import analyze_dimensionality

    print("step 1: COD metadata for metal chalcogenides")
    seen = {}
    for metal in METALS:
        for ch in CUT:
            for lo, hi in ((2, 3), (3, 4), (4, 5)):
                for rec in cod(el1=metal, el2=ch, strictmin=lo, strictmax=hi):
                    f = rec.get("file")
                    if f and f not in seen:
                        rec["_metal"], rec["_chalc"] = metal, ch
                        seen[f] = rec
        print(f"    {metal:>3}  cumulative {len(seen)}")

    print(f"\nstep 2: filtering {len(seen)} entries for an authors' dimensionality statement")
    lab = []
    for r in seen.values():
        L = label_of(r)
        if not L:
            continue
        if elements((r.get("formula") or "")) & COMPETING:
            continue
        r["_dim"], r["_phrase"], r["_field"] = L
        lab.append(r)
    print(f"    labelled, chalcogen-only anion: {len(lab)}")
    dist = {}
    for r in lab:
        dist[r["_dim"]] = dist.get(r["_dim"], 0) + 1
    print("    " + "  ".join(f"{k}D:{v}" for k, v in sorted(dist.items())))

    print(f"\nstep 3: downloading and indexing {len(lab)}")
    rows, fail = [], 0
    for n, r in enumerate(lab, 1):
        cid = r["file"]
        try:
            raw = fetch(f"https://www.crystallography.net/cod/{cid}.cif",
                        os.path.join(CACHE, f"{cid}.cif"))
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            chalcs = sorted(elements(r.get("formula") or "") & set(CUT))
            res = index_on(st, r["_metal"], chalcs)
        except Exception:
            fail += 1
            continue
        if res is None:
            fail += 1
            continue
        rda = None
        try:
            atoms = ase_read(os.path.join(CACHE, f"{cid}.cif"))
            if len(atoms) <= 400:
                a = analyze_dimensionality(atoms, method="RDA")
                rda = a[0].dimtype if a else None
        except Exception:
            pass
        rows.append(dict(cod=cid, metal=r["_metal"], chalcs=chalcs, stated=r["_dim"],
                         phrase=r["_phrase"], rda=rda, nsites=len(st),
                         text=(r.get("title") or r.get("chemname") or "")[:130], **res))
        if n % 25 == 0:
            print(f"    {n}/{len(lab)}  indexed={len(rows)}  failed={fail}")

    json.dump(rows, open(OUT, "w"), indent=1)
    if not rows:
        print("\nno structures indexed")
        return

    agree = sum(1 for r in rows if r["d_top"] == r["stated"])
    print("\n" + "=" * 72)
    print(f"indexed {len(rows)} metal chalcogenides ({fail} skipped)")
    print("=" * 72)
    print(f"  index vs authors' label: {agree}/{len(rows)} = {100.0*agree/len(rows):.1f} per cent")
    print("\n  confusion (rows authors, cols index):")
    print("        " + "".join(f"{c:>7}" for c in (0, 1, 2, 3)))
    for s_ in (0, 1, 2, 3):
        nn = sum(1 for r in rows if r["stated"] == s_)
        if not nn:
            continue
        print(f"   {s_}D  " + "".join(
            f"{sum(1 for r in rows if r['stated']==s_ and r['d_top']==c):>7}" for c in (0, 1, 2, 3))
            + f"   (n={nn})")

    both = [r for r in rows if r["rda"]]
    if both:
        same = sum(1 for r in both if r["rda"] == f"{r['d_top']}D")
        rda_ok = sum(1 for r in both if r["rda"] == f"{r['stated']}D")
        idx_ok = sum(1 for r in both if r["d_top"] == r["stated"])
        print(f"\n  THE KEY COMPARISON, on {len(both)} structures both methods evaluated:")
        print(f"     sublattice and whole network AGREE : {same}/{len(both)}"
              f" = {100.0*same/len(both):.1f} per cent")
        print(f"     whole network vs authors' label    : {rda_ok}/{len(both)}"
              f" = {100.0*rda_ok/len(both):.1f} per cent")
        print(f"     sublattice   vs authors' label     : {idx_ok}/{len(both)}"
              f" = {100.0*idx_ok/len(both):.1f} per cent")
        print("\n  (in the halides the two methods agreed for only 20 per cent;"
              " if they agree here,\n   divergence is a property of the material class,"
              " not of the method)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
