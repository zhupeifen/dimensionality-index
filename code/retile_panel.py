# -*- coding: utf-8 -*-
"""Re-tile a panel's CONTCAR to a different number of repeats.

The CONTCARs under userplot/ were each written as a supercell of the deposited
cell, repeated along the directions that structure's own metal-halide sublattice
propagates along. Changing how many repeats a panel shows is a drawing decision,
not a structural one, so it is done here rather than by rebuilding from the CIF:
the primitive cell is recovered from the file already on disk and re-tiled, which
cannot introduce a different setting, a different atom order, or a different
symmetry reduction than the panel was computed with.

    python retile_panel.py jac_CuI 3 3 2

Writes the new CONTCAR in place, keeping a .bak of the old one. The matching
`box_override` in export_polyhedra.py has to be changed too, or the json the
figure frames each panel with will still describe the old block.
"""
import os
import sys
import io
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_poscar(path):
    L = io.open(path, encoding="utf-8").read().splitlines()
    scale = float(L[1])
    A = np.array([[float(x) for x in L[i].split()] for i in (2, 3, 4)]) * scale
    els = L[5].split()
    ns = [int(x) for x in L[6].split()]
    mode = L[7].strip().lower()
    F = np.array([[float(x) for x in L[8 + i].split()[:3]] for i in range(sum(ns))])
    if mode.startswith(("c", "k")):
        F = F @ np.linalg.inv(A)
    return A, els, ns, F


def write_poscar(path, A, els, ns, F, title):
    out = [title, "1.0"]
    out += ["  %.16f  %.16f  %.16f" % tuple(r) for r in A]
    out += ["  ".join(els), "  ".join(str(n) for n in ns), "Direct"]
    out += ["  %.10f  %.10f  %.10f" % tuple(r) for r in F]
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")


def primitive(A, els, ns, F, old_box):
    """The one-cell contents, given how many times the file repeats it."""
    old_box = np.asarray(old_box)
    Ap = A / old_box[:, None]
    Fp = F * old_box                       # fractional in the primitive cell
    keep = np.all(Fp < 1 - 1e-6, axis=1)   # the atoms of the origin cell
    base, k = {}, 0
    for e, n in zip(els, ns):
        base[e] = Fp[k:k + n][keep[k:k + n]]
        k += n
    return Ap, base


def main():
    if len(sys.argv) not in (5, 8):
        sys.exit("usage: retile_panel.py <folder> <na> <nb> <nc> "
                 "[<old_na> <old_nb> <old_nc>]")
    folder = sys.argv[1]
    new_box = [int(x) for x in sys.argv[2:5]]
    old_box = [int(x) for x in sys.argv[5:8]] if len(sys.argv) == 8 else None

    path = os.path.join(HERE, "userplot", folder, "CONTCAR")
    A, els, ns, F = read_poscar(path)
    print("read %s: %s, cell %s" %
          (path, dict(zip(els, ns)), np.round(np.linalg.norm(A, axis=1), 3)))

    if old_box is None:
        sys.exit("the file does not record how many repeats it holds; pass the "
                 "old box as three more arguments (e.g. 2 2 2)")

    Ap, base = primitive(A, els, ns, F, old_box)
    print("primitive: %s, cell %s" %
          ({e: len(v) for e, v in base.items()},
           np.round(np.linalg.norm(Ap, axis=1), 3)))

    b = np.asarray(new_box)
    An = Ap * b[:, None]
    e2, n2, F2 = [], [], []
    for e, pts0 in base.items():
        pts = np.vstack([(pts0 + [i, j, k]) / b
                         for i in range(b[0]) for j in range(b[1]) for k in range(b[2])])
        e2.append(e)
        n2.append(len(pts))
        F2.append(pts)
    F2 = np.vstack(F2)

    os.replace(path, path + ".bak")
    write_poscar(path, An, e2, n2, F2,
                 "".join("%s%d" % (e, n) for e, n in zip(e2, n2)))
    C = F2 @ An
    print("wrote %s x %s: %s" % (folder, new_box, dict(zip(e2, n2))))
    print("  centre    %s" % np.round((C.max(0) + C.min(0)) / 2, 3))
    print("  half-span %s" % np.round((C.max(0) - C.min(0)) / 2, 3))


if __name__ == "__main__":
    main()
