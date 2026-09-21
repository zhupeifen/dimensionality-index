# -*- coding: utf-8 -*-
"""Complete a local CIF cache for the validation set, by COD entry number.

Structures are not redistributed with this deposit; they are named by
Crystallography Open Database entry number and fetched from COD, which places
its data in the public domain. A snapshot of the COD bulk archive is usually
enough, but entries deposited after the snapshot was taken will be absent from
it: of the 703 halides in validation_guards.json, 58 are missing from a bulk
archive downloaded in 2025 and have to be fetched individually. A reproduction
run against an incomplete cache silently indexes fewer structures and reports
different agreement figures, which is the failure this script exists to prevent.

Reads the entry numbers from a validation file, checks a cache directory in
both the archive's nested layout and a flat <cod>.cif layout, and downloads
whatever is absent.

Usage
    python fetch_cifs.py <validation_guards.json> <cache_dir> [--check]

    --check   report what is missing and exit without downloading
"""
import json
import os
import sys
import time
import urllib.request

URL = "https://www.crystallography.net/cod/%s.cif"
PAUSE = 0.5          # be polite to a free public service


def paths(root, cod):
    """Both layouts a cache may use: the archive's nested one, and flat."""
    return (os.path.join(root, cod[0], cod[1:3], cod[3:5], cod + ".cif"),
            os.path.join(root, cod + ".cif"))


def held(root, cod):
    return any(os.path.exists(p) for p in paths(root, cod))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check" in sys.argv
    if len(args) < 2:
        sys.exit(__doc__)
    src, root = args[0], args[1]

    rows = json.load(open(src))
    if isinstance(rows, dict):
        rows = rows.get("rows", list(rows.values()))
    codes = [r["cod"] for r in rows]
    missing = [c for c in codes if not held(root, c)]

    print("%d entries in %s" % (len(codes), os.path.basename(src)))
    print("%d already in %s" % (len(codes) - len(missing), root))
    print("%d missing" % len(missing))
    if not missing:
        return
    if check_only:
        print("\n".join(missing))
        print("\n--check given; nothing downloaded")
        return

    os.makedirs(root, exist_ok=True)
    got = failed = 0
    for n, cod in enumerate(missing, 1):
        dest = os.path.join(root, cod + ".cif")       # flat; both layouts are read
        try:
            with urllib.request.urlopen(URL % cod, timeout=60) as r:
                data = r.read()
            if not data.strip():
                raise ValueError("empty response")
            open(dest, "wb").write(data)
            got += 1
        except Exception as e:
            failed += 1
            print("   %s failed: %s" % (cod, e))
        if n % 20 == 0:
            print("   %d/%d" % (n, len(missing)))
        time.sleep(PAUSE)

    print("\ndownloaded %d, failed %d" % (got, failed))
    still = [c for c in codes if not held(root, c)]
    print("cache is %s (%d still missing)"
          % ("complete" if not still else "INCOMPLETE", len(still)))


if __name__ == "__main__":
    main()
