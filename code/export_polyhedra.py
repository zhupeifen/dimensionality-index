# -*- coding: utf-8 -*-
"""Coordination polyhedra for the structure panels of the two papers.

The figure has to show what the index measures, so the geometry it draws must be
the geometry the index reads: the same framework metal, the same bridging anion,
the same deposited coordinates. Nothing here is idealised or redrawn by hand.

For each chosen deposition the framework metal sites are replicated over a
display box, each metal's coordinating anions are collected, and the convex hull
of those anions is written out as triangular faces for the surface plus the
subset of hull edges that separate non-coplanar faces, which is what gives a
clean polyhedron outline instead of a triangulated mesh. Spectator cations are
written as sphere centres and the cell as twelve line segments.

Two panel sets are defined. The database paper takes a lead-halide ladder and
the methods paper a copper-halide one, so that neither figure repeats the other:
the same structure drawn in two papers would be a duplicate figure, and the
methods paper is the one going out first.

Usage
    python export_polyhedra.py npj [-o npj_panels.json]
    python export_polyhedra.py jac [-o jac_panels.json]
"""
import argparse
import json
import math
import os
import warnings

warnings.filterwarnings("ignore")

CIF_ROOT = os.path.join("C:\\", "Users", "pzgft", "CODdata", "cif", "cif")

# cod, framework metal, bridging anion, spectator cation, label. A panel may
# have no spectator cation, in which case the structure is the framework alone.
# The display box is not set here: it is chosen from the connectivity, so that
# each panel repeats along the directions the sublattice extends along and along
# no others.
SETS = {
    # the database paper: one lead-halide family, rank 0 to 3
    "npj": [
        dict(cod="1538416", metal="Pb", anion="Br", cation="Cs",
             name="Cs4PbBr6", pretty="Cs_4PbBr_6",
             # The isolated PbBr6 octahedra sit on the cell faces, and the
             # bromides completing them lie up to 0.170, 0.170 and 0.099 of
             # the way outside on a 13.7 x 13.7 x 17.3 A cell. At the 0.06
             # used by default most of them draw open, as loose Pb-Br sticks
             # rather than as the isolated units the panel is there to show.
             batom=(0.185, 0.185, 0.114)),
        dict(cod="4127358", metal="Pb", anion="I", cation="Cs",
             name="CsPbI3-delta", pretty="\delta-CsPbI_3",
             # The margin that would close every octahedron here - 0.031,
             # 0.098, 0.130 - also reaches the neighbouring chains, and they
             # merge into a continuous slab: the 1D panel then reads as 2D.
             # Separation between the chains is what this panel is for, so the
             # default is kept and the boundary octahedra are left to the
             # pruning step instead.
             batom=None),
        dict(cod="9009140", metal="Pb", anion="I", cation=None,
             name="PbI2", pretty="PbI_2",
             # 0.167, 0.167 and 0.133 on a 9.1 x 9.1 x 14.0 A cell. Below
             # this the outer two layers draw without octahedra at all, and
             # a panel that has to show layers is left with one.
             batom=(0.182, 0.182, 0.148)),
        dict(cod="1530681", metal="Pb", anion="Br", cation="Cs",
             name="CsPbBr3", pretty="CsPbBr_3"),
    ],
    # the methods paper: the copper halides it is validated on, rank 0 to 3
    "jac": [
        dict(cod="7246298", metal="Cu", anion="Cl", cation="Cs",
             name="Cs3Cu2Cl5", pretty="Cs_3Cu_2Cl_5"),
        dict(cod="1536279", metal="Cu", anion="I", cation="Cs",
             name="CsCu2I3", pretty="CsCu_2I_3", box_override=(1, 1, 3),
             # Along a, chain axis upright. The automatic rule picks a
             # perpendicular that projects the cell's two double chains onto
             # one another, and the panel reads as a wall; down the chain axis
             # separates them but hides the extension, leaving 1D looking like
             # the 0D panel. This view shows both: chains running the height of
             # the panel with open space between them.
             view_override=(90, 0),
             # wide enough that a Cu at the cell face keeps the iodides that
             # close its tetrahedron. The fraction reaching a 2.6 A bond
             # differs by axis on a 10.5 x 13.1 x 18.2 A cell.
             batom=(0.26, 0.21, 0.15)),
        dict(cod="1528214", metal="Cu", anion="Cl", cation="Rb",
             name="Rb2CuCl4", pretty="Rb_2CuCl_4",
             # Every Cu here is octahedral, and the chlorides completing the
             # octahedra on the cell faces lie outside it: at most 0.150,
             # 0.133 and 0.134 of the way past, on a 15.5 x 14.4 x 14.4 A
             # cell. Below that the outer layers draw with their octahedra
             # open. The values carry a little slack on each.
             batom=(0.165, 0.148, 0.149)),
        dict(cod="7222858", metal="Cu", anion="I", cation=None,
             name="CuI", pretty="CuI", box_override=(2, 2, 2),
             # Two repeats along every axis. A compacter 2x2x1 block was tried,
             # to avoid the 8.6 x 8.6 x 14.2 A column that 2x2x2 gives, and it
             # was a mistake: one period along c is a single slab, and the
             # panel then reads as layers - exactly what the 2D panel beside it
             # shows. A 3D panel has to repeat along all three axes.
             #
             # The camera needs elevation for the same reason. Near edge-on the
             # four tetrahedral layers project onto one another and flatten.
             view_override=(160, 22),
             # 0.167 along a and b reaches the iodides that close a Cu on the
             # cell face; along c nothing is needed, the axial iodide is
             # already inside. Margin along c only adds a further layer of Cu
             # outside the cell with part of their own shell missing, which
             # draw as bare atoms on dangling bonds.
             batom=(0.182, 0.182, 0.02)),
        # CuI is wurtzite-like: a = 4.31, c = 7.09. Two repeats along every
        # axis gives 8.6 x 7.5 x 14.2 A, a narrow column rather than a
        # framework, so c is left at one repeat and the block is near-cubic;
        # the two tetrahedral layers of the stacking are both still present.
        # The wider margin is what makes it read as a framework at all: at the
        # 0.06 used elsewhere only about four of sixteen Cu sites keep the
        # full iodide shell that draws a tetrahedron and the rest come out as
        # bare ball-and-stick. 0.18 is where that saturates; beyond it only
        # loose edge atoms are added.
    ],
}


def cif_path(cod):
    return os.path.join(CIF_ROOT, cod[0], cod[1:3], cod[3:5], cod + ".cif")


def hull_faces_and_edges(pts):
    """Triangular hull faces, plus the hull edges that are real polyhedron edges.

    A hull edge shared by two faces with nearly parallel normals lies inside a
    flat face of the polyhedron and would draw as a diagonal across it, so those
    are dropped and only the creases are kept.
    """
    from scipy.spatial import ConvexHull
    import numpy as np
    h = ConvexHull(np.asarray(pts))
    faces = [list(map(int, s)) for s in h.simplices]
    norms = []
    P = np.asarray(pts)
    cen = P.mean(axis=0)
    fixed = []
    for f in faces:
        a, b, c = P[f[0]], P[f[1]], P[f[2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if np.dot(n, a - cen) < 0:          # outward winding, so lighting works
            f = [f[0], f[2], f[1]]
            n = -n
        fixed.append(f)
        norms.append(n)
    byedge = {}
    for i, f in enumerate(fixed):
        for k in range(3):
            e = tuple(sorted((f[k], f[(k + 1) % 3])))
            byedge.setdefault(e, []).append(i)
    edges = []
    for e, fl in byedge.items():
        if len(fl) != 2 or abs(float(np.dot(norms[fl[0]], norms[fl[1]]))) < 0.999:
            edges.append([e[0], e[1]])
    return fixed, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS), help="which panel set to export")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    panels = SETS[a.set]
    if not a.out:
        a.out = a.set + "_panels.json"

    import numpy as np
    from pymatgen.core import Structure

    idx = {}
    for line in open(r"C:\Users\pzgft\CODdata\cod_dimensionality.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        if r["cod"] in {p["cod"] for p in panels}:
            idx[r["cod"]] = r

    out = []
    for p in panels:
        st = Structure.from_file(cif_path(p["cod"]))
        L = np.asarray(st.lattice.matrix)
        Linv = np.linalg.inv(L)

        base = {}
        for s_ in st:
            base.setdefault(s_.specie.symbol, []).append(
                np.asarray(s_.frac_coords) % 1.0)

        def replicate(species, box, pad=1):
            """Cartesian positions of one species over the box, plus a margin."""
            na, nb, nc = box
            pts = []
            for fr in base.get(species, []):
                for i in range(-pad, na + pad):
                    for j in range(-pad, nb + pad):
                        for k in range(-pad, nc + pad):
                            pts.append((fr + [i, j, k]) @ L)
            return np.asarray(pts)

        # nearest-neighbour cutoff read off the structure itself, not a table
        M1 = replicate(p["metal"], (1, 1, 1))
        X1 = replicate(p["anion"], (1, 1, 1))
        d0 = min(float(np.linalg.norm(m - x)) for m in M1 for x in X1
                 if np.linalg.norm(m - x) > 0.4)
        cut = 1.28 * d0

        def ligands(m, X):
            return [x for x in X if 0.4 < np.linalg.norm(x - m) < cut]

        # Which lattice directions does the sublattice extend along? Bond
        # directions will not answer this: a chain that zigzags has joining
        # vectors with components along two axes while repeating along one. So
        # walk the connected unit instead. Starting from one metal in the centre
        # cell of a 5x5x5 block, step to any metal that shares a coordinating
        # anion, and record which cell each metal reached belongs to. A unit that
        # is periodic along an axis walks out to the block boundary along it; a
        # finite unit stops short.
        R = 5
        half = R // 2
        Mb = replicate(p["metal"], (R, R, R), pad=0)
        Xb = replicate(p["anion"], (R, R, R), pad=0)
        keyed = {}
        for m in Mb:
            keyed[tuple(np.round(m, 3))] = ligands(m, Xb)
        seed = (np.asarray(base[p["metal"]][0]) + [half, half, half]) @ L
        seed = tuple(np.round(seed, 3))
        seen, stack = {seed}, [seed]
        while stack:
            cur = np.asarray(stack.pop())
            lc = keyed[tuple(np.round(cur, 3))]
            for k2, l2 in keyed.items():
                if k2 in seen or np.linalg.norm(np.asarray(k2) - cur) > 2.2 * cut:
                    continue
                if any(np.linalg.norm(x - y) < 0.3 for x in lc for y in l2):
                    seen.add(k2)
                    stack.append(k2)
        cells = np.asarray([np.floor(Linv.T @ np.asarray(k2) + 1e-4)
                            for k2 in seen])
        reach = cells.max(axis=0) - cells.min(axis=0)
        spans = reach >= (R - 1)          # walked the full block along this axis
        nper = int(spans.sum())
        # A chain needs three repeats along itself before it reads as a chain
        # rather than a stack of two; layers and frameworks are unambiguous at
        # two. Panels are framed individually, so a longer 1D panel costs the
        # others nothing.
        mult = 3 if nper == 1 else 2
        box = [int(mult if spans[i] else 1) for i in range(3)]
        if nper == 1:
            # one chain on its own still looks like a solid rod. Repeating once
            # across the chain puts a second chain beside it, and the gap between
            # them is the thing the panel has to show. The shortest perpendicular
            # axis is chosen so the panel stays compact.
            perp = [i for i in range(3) if not spans[i]]
            j = min(perp, key=lambda i: float(np.linalg.norm(L[i])))
            box[j] = 2
        if p.get("box_override"):
            # CsCu2I3 is a double, edge-sharing chain: two of them side by
            # side touch and read as a block. One chain, repeated along
            # itself, shows "extends in one direction only" without the
            # collision. The rule above is right for the other 1D cases.
            box = tuple(p["box_override"])
        else:
            box = tuple(box)

        M = replicate(p["metal"], box)
        X = replicate(p["anion"], box)
        C = replicate(p["cation"], box, pad=0)
        nab = np.asarray(box)

        def inside(cart, eps=1e-6):
            """Half-open box, so a site on a shared face is counted once."""
            f = Linv.T @ cart
            return bool(np.all(f > -eps) and np.all(f < nab - eps))

        polys = []
        for m in M:
            if not inside(m):
                continue
            lig = ligands(m, X)
            if len(lig) < 4:
                continue
            try:
                F, E = hull_faces_and_edges(lig)
            except Exception:
                continue
            polys.append({"V": [[round(float(c), 4) for c in v] for v in lig],
                          "F": [[i + 1 for i in f] for f in F],   # MATLAB is 1-based
                          "E": [[i + 1 for i in e] for e in E],
                          "M": [round(float(c), 4) for c in m]})

        cats = [[round(float(c), 4) for c in v] for v in C if inside(v)]

        na, nb, nc = box
        corners = [(i, j, k) for i in (0, na) for j in (0, nb) for k in (0, nc)]
        segs = []
        for c1 in corners:
            for c2 in corners:
                if sum(1 for u, v in zip(c1, c2) if u != v) == 1 and c1 < c2:
                    segs.append([list(np.asarray(c1, float) @ L),
                                 list(np.asarray(c2, float) @ L)])
        segs = [[[round(float(c), 4) for c in q] for q in s_] for s_ in segs]

        r = idx.get(p["cod"], {})
        out.append({"name": p["name"], "pretty": p["pretty"], "cod": p["cod"],
                    "metal": p["metal"], "anion": p["anion"],
                    "cation": p["cation"] or "",
                    "d_top": r.get("d_top"), "D": r.get("D"),
                    "formula": r.get("formula"),
                    "cn": len(polys[0]["V"]) if polys else 0,
                    "dmin": round(d0, 3), "box": list(box),
                    "L": [[round(float(c), 5) for c in row] for row in L],
                    "spans": [bool(x) for x in spans],
                    "view_override": list(p.get("view_override") or []),
                    "batom": ([float(x) for x in p["batom"]]
                              if isinstance(p.get("batom"), (list, tuple))
                              else float(p.get("batom") or 0)),
                    "polys": polys, "cations": cats, "cell": segs})
        print("{:14s} cod {:>8}  box {}  {:>3} polyhedra  CN {}  {:>3} cations  "
              "d_top {}  D {}  spans {}".format(
                  p["name"], p["cod"], box, len(polys), out[-1]["cn"], len(cats),
                  r.get("d_top"), r.get("D"), nper))

    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote " + a.out)


if __name__ == "__main__":
    main()
