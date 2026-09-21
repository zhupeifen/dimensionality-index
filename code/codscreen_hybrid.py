# -*- coding: utf-8 -*-
"""
Screen hybrid layered copper(II) chlorides for the two-dimensional rung.

Widens the inorganic filter to admit organic A-site cations, which is where the
genuinely Cu-X-Cu-connected layered copper halides live. Reports the index on
the Cu-Cl-Cu network only, so the organic network cannot contribute.

Candidates are ranked by how clean a 2D rung they make: index closest to 2,
fewest elements, and a spacer small enough to be compatible with the
solution routes already established.
"""
import os, sys, json, time, itertools, urllib.request, warnings
import numpy as np

warnings.filterwarnings("ignore")

OUT, CACHE = sys.argv[1], sys.argv[2]
os.makedirs(CACHE, exist_ok=True)

LAM = 1.0
CUT = {"Cl": 3.10, "Br": 3.30, "I": 3.60}
NEXT_CUT = 8.0
ORGANIC = {"C", "N", "H", "O", "S", "F"}          # spacer elements
A_INORG = {"Cs", "Rb", "K", "Na", "Li", "Tl", "NH4"}


def fetch(url, path, pause=0.5):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "dimindex-screen/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    open(path, "wb").write(data)
    time.sleep(pause)
    return data


def parse_formula(f):
    counts = {}
    for tok in f.strip("- ").split():
        sym = "".join(c for c in tok if c.isalpha())
        num = "".join(c for c in tok if c.isdigit())
        if sym:
            counts[sym] = int(num) if num else 1
    return counts


def candidates():
    seen = {}
    for x in ("Cl", "Br"):
        for lo, hi in ((4, 5), (5, 6)):
            url = (f"https://www.crystallography.net/cod/result?el1=Cu&el2={x}&el3=N"
                   f"&strictmin={lo}&strictmax={hi}&format=json")
            try:
                raw = fetch(url, os.path.join(CACHE, f"hyb_Cu_{x}_{lo}{hi}.json"), pause=1.2)
                recs = json.loads(raw.decode("utf-8", "replace"))
            except Exception as e:
                print(f"  query Cu-{x} ({lo}-{hi} elements) failed: {e}")
                continue
            print(f"  Cu-{x}, {lo}-{hi} elements: {len(recs)} records")
            for r in recs:
                f = (r.get("formula") or "").strip("- ").strip()
                c = parse_formula(f)
                els = set(c)
                if "Cu" not in els or x not in els:
                    continue
                other = els - {"Cu", x}
                if not other:
                    continue
                if not other <= (ORGANIC | A_INORG):
                    continue
                if "C" not in els and "N" not in els:      # want a hybrid
                    continue
                # A2CuX4 stoichiometry: halide-to-copper ratio near 4
                if c.get("Cu", 0) == 0:
                    continue
                ratio = c.get(x, 0) / c["Cu"]
                seen[r["file"]] = dict(cod=r["file"], formula=f, halide=x, counts=c,
                                       ratio=round(ratio, 2), sg=r.get("sg"),
                                       title=(r.get("title") or "")[:95],
                                       chemname=(r.get("chemname") or "")[:70])
    return list(seen.values())


def index_on_cu(st, halide, cut):
    A = np.array(st.lattice.matrix)
    frac = np.array([s.frac_coords for s in st.sites]) % 1.0
    sym = [s.specie.symbol for s in st.sites]
    cu = [i for i, e in enumerate(sym) if e == "Cu"]
    xi = [i for i, e in enumerate(sym) if e == halide]
    if not cu or not xi:
        return None
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
    shell = {}
    for i in cu:
        s = set()
        for h in xi:
            d = np.linalg.norm((frac[h][None, :] + imgs - frac[i][None, :]) @ A, axis=1)
            for k in np.where(d <= cut)[0]:
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
                    d = np.linalg.norm((frac[j] + np.array(T) - frac[i]) @ A)
                    br.setdefault((i, j, T), [d, 0])[1] += 1
    if not br:
        return dict(D=0.0, d_top=0, f=0.0, nbridge=0)
    d0 = min(v[0] for v in br.values())
    w = {k: v[1] * np.exp(-(v[0] - d0) / LAM) for k, v in br.items()}

    def rank(bs):
        adj = {}
        for (i, j, T) in bs:
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
                    pos[v] = nv; stack.append(v)
                else:
                    dl = nv - pos[v]
                    if dl.any():
                        trans.append(dl)
        return int(np.linalg.matrix_rank(np.array(trans, float), tol=1e-6)) if trans else 0

    d_top = rank(list(br))
    w_med = float(np.median(list(w.values())))
    imgs2 = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), float)
    have, w_next = set(br), 0.0
    for i in cu:
        for j in cu:
            dd = np.linalg.norm((frac[j][None, :] + imgs2 - frac[i][None, :]) @ A, axis=1)
            for k in np.where((dd > 0.1) & (dd <= NEXT_CUT))[0]:
                T = tuple(imgs2[k].astype(int))
                if (i, j, T) in have:
                    continue
                if rank(list(have) + [(i, j, T)]) > d_top:
                    w_next = max(w_next, float(np.exp(-(dd[k] - d0) / LAM)))
    f = w_next / (w_next + w_med) if (w_next + w_med) > 0 else 0.0
    return dict(D=round(d_top + f, 3), d_top=d_top, f=round(f, 3), nbridge=len(br))


def main():
    from pymatgen.core import Structure
    print("querying COD for hybrid copper halides ...")
    cands = candidates()
    print(f"candidates after filtering: {len(cands)}\n")
    rows, failed = [], 0
    for n, c in enumerate(cands, 1):
        cif = os.path.join(CACHE, f"{c['cod']}.cif")
        try:
            raw = fetch(f"https://www.crystallography.net/cod/{c['cod']}.cif", cif)
            st = Structure.from_str(raw.decode("utf-8", "replace"), fmt="cif")
            res = index_on_cu(st, c["halide"], CUT[c["halide"]])
        except Exception:
            failed += 1
            continue
        if res is None:
            failed += 1
            continue
        rows.append(dict(**c, **res, nsites=len(st)))
        if n % 15 == 0:
            print(f"  {n}/{len(cands)}")
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"\nscreened {len(rows)} ({failed} skipped)\n")

    two = sorted([r for r in rows if r["d_top"] == 2], key=lambda r: (r["nsites"], -r["ratio"]))
    print("=" * 96)
    print(f"TWO-DIMENSIONAL Cu-X-Cu networks: {len(two)} of {len(rows)}")
    print("=" * 96)
    for r in two[:18]:
        print(f"  D={r['D']:<6} X/Cu={r['ratio']:<5} sites={r['nsites']:<4} COD {r['cod']}  "
              f"{r['formula'][:34]:<34} {(r['chemname'] or r['title'])[:44]}")
    dist = {}
    for r in rows:
        dist[r["d_top"]] = dist.get(r["d_top"], 0) + 1
    print(f"\ndimensionality distribution: " +
          "  ".join(f"{k}D:{v}" for k, v in sorted(dist.items())))


if __name__ == "__main__":
    main()
