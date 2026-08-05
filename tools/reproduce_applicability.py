#!/usr/bin/env python3
# Reproduces the "Applicability criterion and operating window" values for
# MSSP-D-26-02329 (HfS2 optical-aging screening) from this public deposit.
#
# WHAT THIS REPRODUCES (paper Section "Applicability criterion and operating window"
# and SI):
#   - Per-condition b* offset-exponential fit  b*(t) = binf + A*exp(-t/tau)
#     on the 33 analysis-query (PNG) images, giving the characteristic time tau.
#   - Saturation time  tsat = tau * ln(A / (2*sigma_pair)).
#   - Local day resolution  sigma_pair / |db*/dt|  at representative ages.
#   - Operating-window gate counts: number of the 21 evolving-condition queries
#     with |b*_query - binf| < 2*sigma_pair, for the robust (0.70) and full-set
#     (2.07) b* single-acquisition scales.
#
# sigma_pair is the within-library single-acquisition b* scale reported in the SI
# (robust value 0.70 b* units, full-set value 2.07 b* units); it is taken here as
# a documented input from the paired-image consistency analysis.
#
# The DB connection is read-only. Run from the repository root (or tools/).
from __future__ import annotations
from pathlib import Path
import sqlite3
import sys

import numpy as np
from scipy.optimize import curve_fit

DB_NAME = "hfs2_oxidation_dataset.db"
OX = ("NativeHfS2-35%RH", "NativeHfS2-70%RH", "PMMA HfS2-70%RH")
SIGMA_ROBUST = 0.70   # SI: RMS(db*)/sqrt(2) over 15 pairs (day-0 N70 format outlier excluded)
SIGMA_FULL = 2.07     # SI: all 16 pairs

# Canonical paper values (revised manuscript + SI)
CANON_TAU = {"NativeHfS2-35%RH": 12.6, "NativeHfS2-70%RH": 3.36, "PMMA HfS2-70%RH": 5.06}
CANON_TSAT = {"NativeHfS2-35%RH": 23, "NativeHfS2-70%RH": 10, "PMMA HfS2-70%RH": 8}
CANON_GATE = {"robust_1.4": 9, "full_4.1": 12}   # of the 21 evolving-condition queries

_npass = _nfail = 0


def check(label, got, want, tol=0.05, fmt=".2f"):
    global _npass, _nfail
    ok = abs(got - want) <= tol
    _npass += ok
    _nfail += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got {got:{fmt}}  (canonical {want:{fmt}}, tol {tol})")


def find_db() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "dbfiles", here.parent / "dbfiles", Path.cwd() / "dbfiles"):
        if (cand / DB_NAME).is_file():
            return cand / DB_NAME
    sys.exit(f"ERROR: dbfiles/{DB_NAME} not found (run from the repository root).")


def model(t, binf, A, tau):
    return binf + A * np.exp(-t / tau)


def main():
    db = find_db()
    print("=" * 78)
    print("MSSP-D-26-02329 applicability-criterion reproduction (b* trajectory fit)")
    print(f"DB: {db}  (read-only)")
    print("=" * 78)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = [
        (str(name), float(day), str(cond), float(b))
        for name, day, cond, b in con.execute(
            "SELECT name, day, cond, lab_b FROM images"
        )
        if b is not None
    ]
    con.close()

    fit = {}
    print("\n## 1. Per-condition b* offset-exponential fit (33 PNG queries)")
    for cond in OX:
        pts = sorted((d, b) for n, d, c, b in rows if c == cond and n.lower().endswith(".png"))
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        binf, A, tau = curve_fit(
            model, t, y, p0=[y.min(), y.max() - y.min(), 5.0],
            maxfev=40000, bounds=([-50, 0, 0.3], [100, 300, 300]),
        )[0]
        fit[cond] = (binf, A, tau)
        check(f"tau {cond}", tau, CANON_TAU[cond])

    print("\n## 2. Saturation time  tsat = tau*ln(A/(2*sigma_pair))  [2*sigma = 1.4]")
    for cond in OX:
        binf, A, tau = fit[cond]
        tsat = tau * np.log(A / (2 * SIGMA_ROBUST))
        check(f"tsat {cond}", tsat, CANON_TSAT[cond], tol=0.6, fmt=".1f")

    print("\n## 3. Native-70%RH local day resolution  sigma_pair/|db*/dt|")
    binf, A, tau = fit["NativeHfS2-70%RH"]
    for t0, want in [(0, 0.1), (7, 0.7), (14, 5.0)]:
        slope = A / tau * np.exp(-t0 / tau)
        check(f"resolution at day {t0}", SIGMA_ROBUST / slope, want, tol=0.6)
    t30 = tau * np.log(A / tau / (SIGMA_ROBUST / 30.0))
    check("age where resolution reaches 30 days (~day 20)", t30, 20.0, tol=1.5, fmt=".1f")

    print("\n## 4. Operating-window gate (|b*_query - binf| < 2*sigma_pair)")
    for key, sig in [("robust_1.4", SIGMA_ROBUST), ("full_4.1", SIGMA_FULL)]:
        n = sum(
            1 for nm, d, c, b in rows
            if c in fit and nm.lower().endswith(".png") and abs(b - fit[c][0]) < 2 * sig
        )
        check(f"gate flagged / 21 ({key})", n, CANON_GATE[key], tol=0.0, fmt="d" if False else ".0f")

    print("\n" + "=" * 78)
    print(f"SUMMARY: {_npass}/{_npass + _nfail} checks PASS, {_nfail} FAIL")
    print("RESULT: " + ("ALL PASS — applicability-section values reproduced from the deposit"
                        if _nfail == 0 else "FAIL"))
    sys.exit(1 if _nfail else 0)


if __name__ == "__main__":
    main()
