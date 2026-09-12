# -*- coding: utf-8 -*-
"""
Index the sublattice dimensionality of every suitable structure in the COD.

Builds the resource behind the planned database paper: for each structure in
the Crystallography Open Database that contains a framework metal bridged by a
single anion class, compute the dimensionality of that metal-anion sublattice.
The quantity is not in any existing database; whole-network dimensionality is,
and Section 5 of the methods paper shows the two differ for most metal halides.

Design notes
  * Reads the bulk CIF tree extracted from cod-cifs-mysql.tgz. No network.
  * Chooses the anion class per structure rather than being told: the halides,
    chalcogens or oxygen, whichever is present and unambiguous. A structure
    containing two competing anion classes is recorded and skipped, because the
    bridging ligand is then not defined.
  * Framework metals are every metal that is not an alkali, alkaline earth or
    ammonium counter-cation, taken together as one sublattice. Evaluating one
    element at a time is wrong for mixed-metal compounds, which is the error
    that cost the chalcogenide run half its agreement.
  * Writes newline-delimited JSON so a run can be resumed and so the output can
    be streamed rather than held in memory.

Usage
    python cod_wide_index.py <cif_root> <out.jsonl> [--limit N] [--workers N]
"""
import itertools
import json
import os
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

LAM = 1.0
NEXT_CUT = 8.0
MAX_METALS = 90                      # the O(N^2) shell search becomes impractical above this

ANION_CUT = {
    "halide":     {"Cl": 3.10, "Br": 3.30, "I": 3.60, "F": 2.70},
    "chalcogen":  {"S": 3.00, "Se": 3.15, "Te": 3.40},
    "oxide":      {"O": 2.80},
}
SPECTATOR = {"Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba", "Ra",
             "H", "C", "N", "P", "B", "Si"}
ALL_ANIONS = set().union(*(set(v) for v in ANION_CUT.values()))


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


def pick_anion_class(present):
    """The anion class to bridge on, or None if it is not well defined."""
    classes = [c for c, cuts in ANION_CUT.items() if present & set(cuts)]
    if len(classes) != 1:
        return None                  # none present, or two classes competing
    return classes[0]


def index_structure(st):
    sym = symbols(st)
    present = {s for s in sym if s}
    cls = pick_anion_class(present)
    if cls is None:
        return None, "anion class not defined"
    cuts = ANION_CUT[cls]
    M = [i for i, e in enumerate(sym) if e and e not in SPECTATOR and e not in ALL_ANIONS]
    X = [(i, sym[i]) for i, e in enumerate(sym) if e in cuts]
    if not M:
        return None, "no framework metal"
    if not X:
        return None, "no bridging anion"
    if len(M) > MAX_METALS:
        return None, "too many metal centres"

    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell = {}
    for i in M:
        s = set()
        for (h, hx) in X:
            d = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(d <= cuts[hx])[0]:
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
        return dict(anion_class=cls, D=0.0, d_top=0, frac=0.0, nbridge=0,
                    nmetal=len(M), nsite=len(sym)), None
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
    return dict(anion_class=cls, D=round(d_top + f, 3), d_top=d_top, frac=round(f, 4),
                nbridge=len(br), nmetal=len(M), nsite=len(sym)), None


FORMULA_RE = re.compile(r"_chemical_formula_sum\s+['\"]?([^'\"\r\n]+)", re.I)
# element with an optional count; a count of zero means the element is absent,
# which COD formula lines record explicitly (e.g. "C8 H20 Bi Cl5 N2 O0") and
# which a bare element match would wrongly read as oxygen being present
ELEM_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def quick_reject(path):
    """Decide from the CIF header whether a full parse is worth it.

    Most of the COD is organic and molecular; parsing and expanding symmetry for
    those costs far more than reading the formula line. Rejecting them here is
    what makes a database-wide run finish in hours rather than a day.
    Returns a reason string to skip, or None to proceed.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(6000)
    except Exception:
        return "unreadable"
    m = FORMULA_RE.search(head)
    if not m:
        return None                      # no formula line: parse it properly
    els = {sym for sym, cnt in ELEM_RE.findall(m.group(1))
           if sym and (cnt == "" or float(cnt) > 0)}
    if not (els - SPECTATOR - ALL_ANIONS):
        return "no framework metal"
    classes = [c for c, cuts in ANION_CUT.items() if els & set(cuts)]
    if len(classes) != 1:
        return "anion class not defined"
    return None


def one_file(path):
    from pymatgen.core import Structure
    cod = os.path.splitext(os.path.basename(path))[0]
    why = quick_reject(path)
    if why:
        return dict(cod=cod, skip=why)
    try:
        st = Structure.from_file(path)
    except Exception:
        return dict(cod=cod, skip="unreadable")
    try:
        res, why = index_structure(st)
    except Exception as e:
        return dict(cod=cod, skip="error:" + type(e).__name__)
    if res is None:
        return dict(cod=cod, skip=why)
    formula = st.composition.reduced_formula
    return dict(cod=cod, formula=formula, **res)


def main():
    root, out = sys.argv[1], sys.argv[2]
    limit = None
    workers = max(1, (os.cpu_count() or 4) - 1)
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["cod"])
                except Exception:
                    pass
        print(f"resuming: {len(done)} already indexed")

    print(f"scanning {root} for CIF files ...")
    files = []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.endswith(".cif") and os.path.splitext(n)[0] not in done:
                files.append(os.path.join(dirpath, n))
                if limit and len(files) >= limit:
                    break
        if limit and len(files) >= limit:
            break
    print(f"to process: {len(files)}   workers: {workers}")

    t0 = time.time()
    n = ok = 0
    with open(out, "a", encoding="utf-8") as fh, \
         ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_file, f): f for f in files}
        for fut in as_completed(futs):
            n += 1
            try:
                rec = fut.result()
            except Exception:
                continue
            if "skip" not in rec:
                ok += 1
            fh.write(json.dumps(rec) + "\n")
            if n % 2000 == 0:
                el = time.time() - t0
                fh.flush()
                print(f"   {n}/{len(files)}  indexed={ok}  "
                      f"{n/el:.0f}/s  eta {(len(files)-n)/max(1e-9, n/el)/60:.0f} min")
    print(f"\nfinished: {n} processed, {ok} indexed, {time.time()-t0:.0f} s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
