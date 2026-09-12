# -*- coding: utf-8 -*-
"""
Build an external validation set for the dimensionality index.

The validation in the manuscript uses structures we selected ourselves, which is
the standard objection to a new descriptor. This script builds a set we did not
choose and did not label: crystal structures whose depositing authors state the
dimensionality of the inorganic network in the publication title or chemical
name recorded in the Crystallography Open Database.

Procedure
  1. query COD metadata for metal-halide structures (cheap, no CIF download)
  2. keep only entries whose title or chemical name states a dimensionality
     unambiguously, and record the phrase that produced the label
  3. download and index only those, on the metal-halide sublattice
  4. report agreement, and report the failure modes honestly

The label is the authors' claim, not ground truth: some are wrong, and some
describe the organic packing rather than the inorganic network. Disagreements
are therefore written out in full for manual reading rather than being counted
as errors. That list is itself a result.

Usage
    python harvest_validation.py <out.json> [cache_dir]
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
CACHE = sys.argv[2] if len(sys.argv) > 2 else "codcache_validation"
os.makedirs(CACHE, exist_ok=True)

LAM = 1.0
NEXT_CUT = 8.0
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}
METALS = ["Pb", "Sn", "Bi", "Sb", "Cu", "Ag", "Mn", "Cd"]
HALIDES = ["Cl", "Br", "I"]

# Phrases that state a dimensionality. Ordered: the first match wins, so the
# more specific phrasings are tested before the looser ones.
LABELS = [
    (3, r"\bthree[- ]dimensional\b|\b3D\b(?!\s*material)"),
    (2, r"\btwo[- ]dimensional\b|\b2D\b(?!\s*material)|\blayered\b|\bbilayer(ed)?\b"),
    (1, r"\bone[- ]dimensional\b|\b1D\b|\bchain[- ]like\b|\bwire[- ]like\b"),
    (0, r"\bzero[- ]dimensional\b|\b0D\b|\bisolated\b.{0,20}\b(cluster|unit|octahedra|tetrahedra)"),
]
# Phrases that make a label unusable: they qualify or negate the dimensionality,
# or they describe a different network from the one being indexed.
VETO = re.compile(
    r"quasi[- ]|pseudo[- ]|from\s+\dD\s+to\s+\dD|dimensional\s+reduction|"
    r"\bhybrid\s+\dD\b.{0,12}\bframework\b|organic\s+(layer|network|framework)",
    re.I,
)


def fetch(url, path, pause=0.5, timeout=90):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "dimindex-validation/1.0"})
    data = urllib.request.urlopen(req, timeout=timeout).read()
    open(path, "wb").write(data)
    time.sleep(pause)
    return data


def cod_query(**kw):
    url = "https://www.crystallography.net/cod/result?" + urllib.parse.urlencode(
        dict(kw, format="json"))
    key = "q_" + re.sub(r"\W+", "_", str(sorted(kw.items())))[:90] + ".json"
    try:
        return json.loads(fetch(url, os.path.join(CACHE, key), pause=1.0)
                          .decode("utf-8", "replace"))
    except Exception as e:
        print(f"    query failed {kw}: {e}")
        return []


def label_of(rec):
    """Return (dimensionality, phrase, field) if the authors state one."""
    for field in ("title", "chemname"):
        text = (rec.get(field) or "").strip()
        if not text or VETO.search(text):
            continue
        for dim, pat in LABELS:
            m = re.search(pat, text, re.I)
            if m:
                return dim, m.group(0), field
    return None


def symbols(st):
    """Element per site, tolerating partial occupancy; flags disorder."""
    out, disordered = [], False
    for s in st.sites:
        try:
            out.append(s.specie.symbol)
        except Exception:
            disordered = True
            try:
                out.append(max(s.species, key=s.species.get).symbol)
            except Exception:
                out.append(None)
    return out, disordered


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


def index_on(st, metal, halide, cut):
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym, disordered = symbols(st)
    M = [i for i, e in enumerate(sym) if e == metal]
    X = [i for i, e in enumerate(sym) if e == halide]
    if not M or not X:
        return None, disordered
    if len(M) > 90:                      # keep the O(N^2) shell search bounded
        return "toobig", disordered
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell = {}
    for i in M:
        s = set()
        for h in X:
            d = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(d <= cut)[0]:
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
        return dict(D=0.0, d_top=0, frac=0.0, nbridge=0), disordered
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
                    cand = float(np.exp(-(dd[k] - d0) / LAM))
                    if cand > wnext:
                        wnext, dnext = cand, float(dd[k])
    f = wnext / (wnext + wmed) if (wnext + wmed) > 0 else 0.0
    return (dict(D=round(d_top + f, 3), d_top=d_top, frac=round(f, 4),
                 nbridge=len(br), d_next=None if dnext is None else round(dnext, 3),
                 d0=round(d0, 3)), disordered)


def main():
    from pymatgen.core import Structure

    print("step 1: querying COD metadata (no CIF downloads yet)")
    seen = {}
    for metal in METALS:
        for hal in HALIDES:
            for extra in ({}, {"el3": "N"}):
                for lo, hi in ((2, 3), (3, 4), (4, 5), (5, 6)):
                    recs = cod_query(el1=metal, el2=hal, strictmin=lo, strictmax=hi, **extra)
                    for r in recs:
                        f = r.get("file")
                        if f and f not in seen:
                            r["_metal"], r["_halide"] = metal, hal
                            seen[f] = r
            print(f"    {metal:>3}-{hal:<3} cumulative unique entries: {len(seen)}")

    print(f"\nstep 2: filtering {len(seen)} entries for an authors' dimensionality statement")
    labelled = []
    for r in seen.values():
        lab = label_of(r)
        if lab:
            dim, phrase, field = lab
            r["_dim"], r["_phrase"], r["_field"] = dim, phrase, field
            labelled.append(r)
    print(f"    entries carrying an explicit label: {len(labelled)}")
    by_dim = {}
    for r in labelled:
        by_dim[r["_dim"]] = by_dim.get(r["_dim"], 0) + 1
    print("    " + "  ".join(f"{k}D:{v}" for k, v in sorted(by_dim.items())))

    print(f"\nstep 3: downloading and indexing {len(labelled)} structures")
    rows, fails = [], {"read": 0, "disordered_read": 0, "no_network": 0, "toobig": 0}
    for n, r in enumerate(labelled, 1):
        cod = r["file"]
        try:
            raw = fetch(f"https://www.crystallography.net/cod/{cod}.cif",
                        os.path.join(CACHE, f"{cod}.cif"))
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
        except Exception:
            fails["read"] += 1
            continue
        try:
            res, disordered = index_on(st, r["_metal"], r["_halide"], CUT[r["_halide"]])
        except Exception:
            fails["disordered_read"] += 1
            continue
        if res is None:
            fails["no_network"] += 1
            continue
        if res == "toobig":
            fails["toobig"] += 1
            continue
        rows.append(dict(cod=cod, metal=r["_metal"], halide=r["_halide"],
                         stated=r["_dim"], phrase=r["_phrase"], field=r["_field"],
                         text=((r.get("title") or r.get("chemname") or "")[:150]),
                         formula=(r.get("formula") or "").strip("- "),
                         disordered=bool(disordered), nsites=len(st), **res))
        if n % 25 == 0:
            print(f"    {n}/{len(labelled)}  indexed={len(rows)}  failed={sum(fails.values())}")

    json.dump(dict(rows=rows, fails=fails, n_queried=len(seen), n_labelled=len(labelled)),
              open(OUT, "w"), indent=1)

    agree = [r for r in rows if r["d_top"] == r["stated"]]
    print("\n" + "=" * 78)
    print(f"indexed {len(rows)} structures carrying an authors' dimensionality label")
    print("=" * 78)
    print(f"  agreement with the authors' own label: {len(agree)}/{len(rows)}"
          f"  ({100.0*len(agree)/max(1,len(rows)):.1f} per cent)")
    print(f"  failures: {fails}")
    print("\n  confusion (rows = stated by authors, cols = index):")
    print("        " + "".join(f"{c:>7}" for c in (0, 1, 2, 3)))
    for s_ in (0, 1, 2, 3):
        line = f"    {s_}D  "
        for c in (0, 1, 2, 3):
            line += f"{sum(1 for r in rows if r['stated']==s_ and r['d_top']==c):>7}"
        print(line)
    print(f"\n  fractional part: nonzero in "
          f"{sum(1 for r in rows if r['frac'] > 0.001)}/{len(rows)} structures")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
