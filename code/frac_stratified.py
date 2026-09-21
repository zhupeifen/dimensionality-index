# -*- coding: utf-8 -*-
"""Does a nonzero fractional term mark a contested assignment, or only a rank?

The headline comparison in Section 3 - agreement 83.8 per cent where the
fractional term is zero against 33.3 per cent where it is nonzero - pools
structures of different computed rank. Rank is itself a strong predictor of
agreement, and a nonzero term is impossible at rank three, so the two groups
cannot have the same rank composition. This script separates the two effects by
stratifying on computed rank, which is what the manuscript now reports.

Structures with no metal-ligand-metal bridge are held out: the term is
undefined for them, not zero.

Usage
    python frac_stratified.py [validation_allmetal703.json]
"""
import collections
import json
import os
import sys
from math import comb, erfc, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "validation_allmetal703.json")


def fisher_two_sided(ok0, n0, ok1, n1):
    """Exact p for the 2x2 table [ok0, n0-ok0; ok1, n1-ok1]."""
    k = ok0 + ok1
    lo, hi = max(0, k - n1), min(n0, k)
    tot = sum(comb(n0, i) * comb(n1, k - i) for i in range(lo, hi + 1))
    p_obs = comb(n0, ok0) * comb(n1, k - ok0) / tot
    return sum(comb(n0, i) * comb(n1, k - i) / tot for i in range(lo, hi + 1)
               if comb(n0, i) * comb(n1, k - i) / tot <= p_obs * (1 + 1e-9))


def main():
    rows = json.load(open(SRC))
    defined = [r for r in rows if (r.get("nbridge") or 0) > 0]
    print("%d structures, %d with a bridging network, %d without"
          % (len(rows), len(defined), len(rows) - len(defined)))

    tab = collections.defaultdict(lambda: [0, 0, 0, 0])   # ok|f=0, n|f=0, ok|f>0, n|f>0
    for r in defined:
        i = 2 if (r.get("frac") or 0) != 0 else 0
        tab[r["d_top"]][i + 1] += 1
        tab[r["d_top"]][i] += int(r["d_top"] == r["stated"])

    print("\n rank |        f = 0        |        f > 0        |  Fisher p")
    tot = [0, 0, 0, 0]
    for k in sorted(tab):
        a, b, c, d = tab[k]
        tot = [tot[i] + tab[k][i] for i in range(4)]
        s0 = "%3d/%-3d = %5.1f%%" % (a, b, 100.0 * a / b) if b else "        -        "
        s1 = "%3d/%-3d = %5.1f%%" % (c, d, 100.0 * c / d) if d else "  impossible     "
        p = "  %.4f" % fisher_two_sided(a, b, c, d) if (b and d) else "     -"
        print("  %dD  | %s | %s |%s" % (k, s0, s1, p))
    print(" pool | %3d/%-3d = %5.1f%% | %3d/%-3d = %5.1f%% |"
          % (tot[0], tot[1], 100.0 * tot[0] / tot[1],
             tot[2], tot[3], 100.0 * tot[2] / tot[3]))

    # Mantel-Haenszel, over the strata in which a nonzero term is possible at
    # all. The odds ratio says how large the association is once rank is
    # controlled; the chi-squared says whether what survives is more than noise.
    num = den = S = V = 0.0
    for k in sorted(tab):
        a, b, c, d = tab[k]
        n = b + d
        if not (b and d):
            continue
        num += a * (d - c) / n
        den += c * (b - a) / n
        m1 = a + c
        S += a - b * m1 / n
        V += b * d * m1 * (n - m1) / (n * n * (n - 1))
    chi = S * S / V
    chi_c = (abs(S) - 0.5) ** 2 / V
    print("\nMantel-Haenszel odds ratio for agreement, rank held fixed: %.2f" % (num / den))
    print("Mantel-Haenszel chi-squared: %.2f on 1 d.f., p = %.4f "
          "(continuity-corrected %.2f, p = %.4f)"
          % (chi, erfc(sqrt(chi / 2)), chi_c, erfc(sqrt(chi_c / 2))))
    print("Pooled unadjusted odds ratio: %.2f"
          % ((tot[0] / (tot[1] - tot[0])) / (tot[2] / (tot[3] - tot[2]))))


if __name__ == "__main__":
    main()
