# -*- coding: utf-8 -*-
"""
Electronic dimensionality index D from a VASP CONTCAR.

D = d_top + f

  d_top : integer periodic rank of the metal-halide bridging network
          (rank of the lattice-translation subgroup under which the
          connected component maps onto itself).  0 = isolated cluster,
          1 = chain, 2 = layer, 3 = framework.

  f     : continuous inter-component coupling fraction,
          f = w_next / (w_next + w_med)   in [0, 1)
          w_next = strongest metal-metal coupling that would raise the rank,
                  taken no closer than d0 (see the guard in next_coupling)
          w_med  = median coupling inside the network

Coupling weight for a metal-metal pair bridged by a shared halide:
  w = m * exp(-(d - d0) / lam)
  m   = number of shared bridging halides (bridging multiplicity)
  d   = metal-metal distance
  d0  = shortest bridged metal-metal distance in the structure
  lam = 1.0 A decay length
"""
import sys, json, itertools
import numpy as np

LAM = 1.0          # A, coupling decay length
CUT_MX = 3.10      # A, metal-halide first-shell cutoff
CUT_NEXT = 8.0     # A, search radius for rank-raising couplings


def read_contcar(path):
    with open(path, "r", errors="replace") as fh:
        L = [ln.rstrip("\n") for ln in fh]
    scale = float(L[1].split()[0])
    A = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]) * scale
    species = L[5].split()
    counts = [int(x) for x in L[6].split()]
    mode = L[7].strip().lower()
    nat = sum(counts)
    coords = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(nat)])
    if mode.startswith("d"):
        frac = coords
    else:
        frac = coords @ np.linalg.inv(A)
    labels = []
    for sp, c in zip(species, counts):
        labels += [sp] * c
    return A, np.array(labels), frac % 1.0


def bridges(A, labels, frac, metals, halide="Cl", cut=CUT_MX):
    """Return list of (i, j, T, d, nshared) for metal i in cell 0 bridged to
    metal j in cell T through >=1 shared halide."""
    mi = [k for k, s in enumerate(labels) if s in metals]
    xi = [k for k, s in enumerate(labels) if s == halide]
    imgs = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)

    # metal -> set of (halide_index, halide_image) within cutoff
    coord = {}
    for i in mi:
        shell = set()
        for h in xi:
            dfr = frac[h][None, :] + imgs - frac[i][None, :]
            dcart = dfr @ A
            dist = np.linalg.norm(dcart, axis=1)
            for k in np.where(dist <= cut)[0]:
                shell.add((h, tuple(imgs[k].astype(int))))
        coord[i] = shell

    # two metals bridge if their halide shells intersect (same halide, same image)
    out = {}
    for i in mi:
        for j in mi:
            for (h, ti) in coord[i]:
                # halide (h, ti) as seen from i; which metals touch that same halide?
                for (h2, tj) in coord[j]:
                    if h2 != h:
                        continue
                    # j sits at cell (ti - tj) relative to i
                    T = tuple(np.array(ti) - np.array(tj))
                    if i == j and T == (0, 0, 0):
                        continue
                    d = np.linalg.norm((frac[j] + np.array(T) - frac[i]) @ A)
                    key = (i, j, T)
                    if key not in out:
                        out[key] = [d, 0]
                    out[key][1] += 1
    return [(i, j, T, v[0], v[1]) for (i, j, T), v in out.items()]


def periodic_rank(br, metals_idx):
    """Rank of lattice translations that map the connected component to itself."""
    adj = {}
    for (i, j, T, d, m) in br:
        adj.setdefault(i, []).append((j, np.array(T)))
    if not adj:
        return 0, []
    start = sorted(adj.keys())[0]
    pos = {start: np.zeros(3, int)}
    stack = [start]
    trans = []
    while stack:
        u = stack.pop()
        for (v, T) in adj.get(u, []):
            nv = pos[u] + T
            if v not in pos:
                pos[v] = nv
                stack.append(v)
            else:
                delta = nv - pos[v]
                if delta.any():
                    trans.append(delta)
    if not trans:
        return 0, []
    M = np.array(trans, dtype=float)
    rank = np.linalg.matrix_rank(M, tol=1e-6)
    return int(rank), trans


def analyse(path, metals, halide="Cl"):
    A, labels, frac = read_contcar(path)
    br = bridges(A, labels, frac, metals, halide)
    if not br:
        return dict(D=0.0, d_top=0, f=0.0, nbridge=0,
                    formula=formula(labels), note="no bridging halides")
    d0 = min(b[3] for b in br)
    w = {}
    for (i, j, T, d, m) in br:
        w[(i, j, T)] = m * np.exp(-(d - d0) / LAM)
    d_top, _ = periodic_rank(br, None)
    w_med = float(np.median(list(w.values())))

    # strongest coupling that would raise the rank: search longer metal-metal
    # contacts not already in the bridged network
    mi = [k for k, s in enumerate(labels) if s in metals]
    imgs = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), dtype=float)
    have = {(i, j, T) for (i, j, T, d, m) in br}
    w_next = 0.0
    for i in mi:
        for j in mi:
            dfr = frac[j][None, :] + imgs - frac[i][None, :]
            dist = np.linalg.norm(dfr @ A, axis=1)
            for k in np.where((dist >= d0) & (dist <= CUT_NEXT))[0]:
                # dist >= d0: a connection closer than the nearest bridged pair is
                # not a next-dimension connection. Without this guard a direct
                # metal-metal contact inside a cluster (cuprophilic Cu-Cu at 2.5 A
                # against a shortest bridged separation of 5.4 A) inverts the
                # exponential and returns a spurious fractional part near unity.
                T = tuple(imgs[k].astype(int))
                if (i, j, T) in have:
                    continue
                trial = br + [(i, j, T, dist[k], 1)]
                r2, _ = periodic_rank(trial, None)
                if r2 > d_top:
                    w_next = max(w_next, np.exp(-(dist[k] - d0) / LAM))
    f = w_next / (w_next + w_med) if (w_next + w_med) > 0 else 0.0
    return dict(D=round(d_top + f, 3), d_top=d_top, f=round(f, 3),
                nbridge=len(br), d0=round(d0, 3), w_med=round(w_med, 4),
                w_next=round(float(w_next), 6), formula=formula(labels))


def formula(labels):
    u, c = np.unique(labels, return_counts=True)
    order = {"Cs": 0, "Cu": 1, "Sb": 2, "Cl": 3}
    pairs = sorted(zip(u, c), key=lambda t: order.get(t[0], 9))
    return "".join(f"{s}{n}" for s, n in pairs)


if __name__ == "__main__":
    base = sys.argv[1]
    cases = [
        ("126-0D-Cl", "Cs3Cu2Cl5",      "0D (nominal)", ["Cu"]),
        ("278-1D-Cl", "CsCu2Cl3",       "1D (nominal)", ["Cu"]),
        ("292-2D-Cl", "Cs8Cu2Sb4Cl24",  "2D (nominal)", ["Cu"]),
        ("292-2D-Cl", "Cs8Cu2Sb4Cl24",  "2D (Cu+Sb)",   ["Cu", "Sb"]),
        ("Cs2CuCl4",  "Cs2CuCl4",       "0D (Cu II)",   ["Cu"]),
        ("CsCuCl3",   "CsCuCl3",        "1D (Cu II)",   ["Cu"]),
    ]
    rows = []
    for folder, nominal, tag, metals in cases:
        p = f"{base}/{folder}/CONTCAR"
        try:
            r = analyse(p, metals)
        except Exception as e:
            r = dict(D=None, note=f"ERROR {e}")
        r.update(folder=folder, nominal=nominal, tag=tag, metals="+".join(metals))
        rows.append(r)
        print(f"{tag:16s} {r.get('formula','?'):18s} metals={r['metals']:6s} "
              f"D={r.get('D')}  d_top={r.get('d_top')}  f={r.get('f')}  "
              f"bridges={r.get('nbridge')}  d0={r.get('d0')}")
    with open(sys.argv[2], "w") as fh:
        json.dump(rows, fh, indent=2)
