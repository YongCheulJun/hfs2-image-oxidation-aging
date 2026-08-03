#!/usr/bin/env python3
# Self-contained script that reproduces and verifies the canonical MSSP-D-26-02329
# values (manual-ROI LODO + A1g regression) from the public deposit DB alone.
"""
Canonical-results reproduction script for MSSP-D-26-02329 (HfS2 optical degradation).
================================================================================

WHAT THIS REPRODUCES (paper values, manual-ROI canonical pipeline):
  1. Data structure ..... 33 PNG (queries) + 20 JPG (reference library) = 53 images,
                          16 exact (condition, day) overlaps.
  2. LODO evaluation .... reference = 20 JPG, query = 33 PNG, leave-out = every JPG
                          sharing the query's exact (condition, day); headline metric =
                          pooled RMSE over the 3 oxidizing conditions (n = 21).
                          b*+clip 6.78 / kNN 7.70 / ensemble 7.94 / spatial 9.95 /
                          Wasserstein 8.77 / FFT 12.78 / pool-mean 9.84 (days).
  3. Ensemble weights ... Huber fit on ox-21: kNN 0.91 / spatial 0.02 / Wass 0.07 / FFT 0.
  4. Significance ....... b*+clip vs pool-mean: dRMSE +3.06, bootstrap 95% CI
                          [+1.55, +4.42], Wilcoxon two-sided p = 0.0030, 18/21 wins.
                          kNN vs pool-mean: CI [+0.15, +4.78] (excludes 0).
  5. A1g regression ..... 4-OLS R^2-weighted LOO on the 20 Raman-linked JPG
                          (condition, day) pairs — the images that physically carry
                          the paired Raman spectra — with the STORED dE (JPG own-pool
                          day-0 reference): R^2 = 0.600, RMSE = 0.199, r = 0.794,
                          MAE = 0.159.
  6. Descriptor facts ... N70 b* day0 = 33.3 -> day14 = 3.0; r(b*,S), r(b*,YI),
                          r(S,YI) >= 0.997 over the 33 PNG; PNG stored delta_e is
                          same-pool (reference = day-0 of the same condition), hence
                          exactly 0 at day 0 and reproducible from Lab alone.
  7. Table 1 decay ...... (1 - A1g(28d)) x 100 = 60.1 / 94.0 / 13.5 / 47.3 %.

DATA SOURCES (this repository only — nothing else is read):
  dbfiles/hfs2_oxidation_dataset.db
                        — 53 images (33 PNG queries + 20 JPG reference library):
                        RGB blob, saved MANUAL ROI (roi_x0..roi_y1), the descriptors
                        (L*, a*, b*, S, YI) computed on it, and the same-pool delta_e
                        (reference = day-0 of the same condition; day-0 dE = 0).
  dataset/raman_a1g_values.csv
                        — Ref. [13] published normalized A1g per (condition, day),
                        deposited as data (single source of truth; the script loads
                        it, nothing is hardcoded). Raw Raman spectra are NOT
                        redistributed and NOT used (A1g decays come from Ref. [13]).
  All DB connections are read-only (?mode=ro); nothing is written to the repo.

PIXEL-FIDELITY GATE (§0):
  Before any headline number is checked, the stored colour descriptors (L*, a*, b*,
  S, YI) are RE-COMPUTED from the raw rgb_blob + saved ROI and asserted equal to the
  stored values. This makes the headline metrics a function of the pixels, not merely
  a read-back of stored numbers — corrupt the image and the gate fails.

IMPORTANT — ROI POLICY:
  Every computation uses the SAVED manual ROI stored in the DB (roi_x0..roi_y1)
  and the descriptors stored with it. auto_detect_roi is NOT used anywhere; this
  script does not import the pipeline code (hfs2_v5_49.py) at all.

CLIP NOTE: the b* regression prediction is clipped to [0, 29] days; on this data
  only the lower bound (0) ever activates.

HOW TO RUN:
  python3 reproduce_canonical.py        (from the repository root, or leave the
                                         script in the repo root / tools/)
  Requires: numpy, scipy, opencv-python, Pillow.
  Every canonical value is recomputed and asserted (PASS if within +-0.05 unless a
  tighter/exact tolerance applies). Exit code 0 = all PASS.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- configuration
OX = ["NativeHfS2-35%RH", "NativeHfS2-70%RH", "PMMA HfS2-70%RH"]  # oxidizing conds
CONDS = ["NativeHfS2-35%RH", "NativeHfS2-70%RH", "Al2O3HfS2-70%RH", "PMMA HfS2-70%RH"]
RAMAN_DAYS = [0, 3, 7, 14, 28]
HUBER_DELTA = 5.0
CLIP_LO, CLIP_HI = 0.0, 29.0
BOOT_B, BOOT_SEED = 20000, 0

# Ref.[13] published normalized A1g — deposited as data, NOT hardcoded, so the values
# can be audited independently of this code (single source of truth). Raw Raman spectra
# are NOT redistributed; only these Ref.[13] per-(condition, day) values are used. Any
# silent edit to the CSV makes the A1g R^2 (§8) and decay % (§9) checks FAIL.
def _load_measured_a1g() -> dict[str, list[float]]:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "dataset", here / "dataset", Path.cwd() / "dataset"):
        p = cand / "raman_a1g_values.csv"
        if p.is_file():
            break
    else:
        sys.exit("ERROR: dataset/raman_a1g_values.csv not found (run from the repository root).")
    out = {c: [None] * len(RAMAN_DAYS) for c in CONDS}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["condition"]][RAMAN_DAYS.index(int(row["day"]))] = float(row["a1g_normalized"])
    if any(v is None for vals in out.values() for v in vals):
        sys.exit("ERROR: dataset/raman_a1g_values.csv is missing (condition, day) rows.")
    return out


MEASURED_A1G = _load_measured_a1g()


DB_NAME = "hfs2_oxidation_dataset.db"


def find_dbdir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "dbfiles", here.parent / "dbfiles", Path.cwd() / "dbfiles"):
        if (cand / DB_NAME).is_file():
            return cand
    sys.exit(f"ERROR: dbfiles/{DB_NAME} not found (run from the repository root).")


# ------------------------------------------------------------- PASS/FAIL ledger
CHECKS: list[tuple[str, bool]] = []


def check(name: str, got: float, want: float, tol: float = 0.05, fmt: str = ".2f") -> None:
    ok = np.isfinite(got) and abs(got - want) <= tol
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:{fmt}}  (canonical {want:{fmt}}, tol {tol})")


def check_int(name: str, got: int, want: int) -> None:
    ok = int(got) == int(want)
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}  (canonical {want}, exact)")


def check_true(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")


# ------------------------------------------------------------------ data access
def load_images(db: Path) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT name, day, cond, roi_x0, roi_y0, roi_x1, roi_y1, "
        "s_mean, yellowness_idx, lab_L, lab_a, lab_b, delta_e, rgb_blob FROM images"
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        rgb = np.array(Image.open(io.BytesIO(r[13])).convert("RGB"))
        out.append(dict(name=r[0], day=float(r[1]), cond=r[2], roi=(r[3], r[4], r[5], r[6]),
                        s=float(r[7]), yi=float(r[8]), L=float(r[9]), a=float(r[10]),
                        b=float(r[11]), de=float(r[12]), rgb=rgb))
    return out


# --------------------------------------------------- features (saved-ROI based)
def roi_mask(shape, roi):
    m = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = roi
    m[y0:y1, x0:x1] = True
    return m


def bstar_map(rgb):
    lab = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab).astype(np.float32)
    return lab[:, :, 2] - 128.0


# --- pixel-fidelity recomputation (identical formulas to the analysis pipeline) ---
def recompute_lab(rgb, m):
    lab = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab).astype(np.float32)
    return (float(np.mean(lab[:, :, 0][m] / 255.0 * 100.0)),
            float(np.mean(lab[:, :, 1][m] - 128.0)),
            float(np.mean(lab[:, :, 2][m] - 128.0)))


def recompute_s_mean(rgb, m):
    img = rgb.astype(np.float64) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    I = (R + G + B) / 3.0
    mn = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 1e-8, 1.0 - mn / (I + 1e-10), 0.0)
    return float(np.mean((S * 255).clip(0, 255).astype(np.uint8)[m]))


def recompute_yi(rgb, m):
    img = rgb.astype(np.float64) / 255.0
    R = img[:, :, 0][m]; G = img[:, :, 1][m]; B = img[:, :, 2][m]
    return float(np.mean(100.0 * (1.28 * R - 1.06 * B) / np.where(G > 0.01, G, 0.01)))


def hist_sig(rgb, m, bins=64):
    b = bstar_map(rgb)[m]
    h, _ = np.histogram(b, bins=bins, range=(-30.0, 80.0))
    t = h.sum()
    return (np.ones(bins) / bins) if t == 0 else h.astype(np.float64) / t


def fft_feat(rgb, m):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mv = float(gray[m].mean())
    g = np.full_like(gray, mv)
    g[m] = gray[m]
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    hf = P[r_map > max_r * 0.40].sum()
    tot = P[r_map > max_r * 0.02].sum()
    hf_ratio = float(hf / (tot + 1e-12))
    edges = np.linspace(0, max_r, 65)
    radial = np.zeros(64)
    for i in range(64):
        ring = (r_map >= edges[i]) & (r_map < edges[i + 1])
        if ring.sum() > 0:
            radial[i] = float(P[ring].mean())
    radial = radial / (radial.sum() + 1e-12)
    ent = float(-np.sum(radial * np.log(radial + 1e-12)))
    return hf_ratio, ent, radial


def spatial_feat(rgb, m, roi, rows=3, cols=3):
    b = bstar_map(rgb)
    x0, y0, x1, y1 = roi
    sh, sw = (y1 - y0) / rows, (x1 - x0) / cols
    seg = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            ry0 = y0 + int(r * sh); ry1 = y0 + int((r + 1) * sh) if r < rows - 1 else y1
            rx0 = x0 + int(c * sw); rx1 = x0 + int((c + 1) * sw) if c < cols - 1 else x1
            vals = b[ry0:ry1, rx0:rx1][m[ry0:ry1, rx0:rx1]]
            if len(vals) >= 5:
                seg[r, c] = float(np.mean(vals))
    ent = float(np.nanstd(seg))
    # original definition: 3x3 center block covers all -> boundary empty -> grad = 0
    cr, cc = rows // 2, cols // 2
    cm = np.zeros((rows, cols), bool)
    cm[max(0, cr - 1):cr + 2, max(0, cc - 1):cc + 2] = True
    bd = ~cm
    cv_ = seg[cm & ~np.isnan(seg)]; bv = seg[bd & ~np.isnan(seg)]
    grad = float(np.mean(bv) - np.mean(cv_)) if (len(cv_) and len(bv)) else 0.0
    rv = [np.nanvar(seg[r, :]) for r in range(rows) if not np.all(np.isnan(seg[r, :]))]
    cvv = [np.nanvar(seg[:, c]) for c in range(cols) if not np.all(np.isnan(seg[:, c]))]
    rvar = float(np.mean(rv)) if rv else 0.0
    cvar = float(np.mean(cvv)) if cvv else 0.0
    ani = rvar / (cvar + 1e-6) if cvar > 1e-6 else 1.0
    return ent, grad, ani


# ------------------------------------------------------------------- estimators
def top3_est(scores):  # [(dist, day)] -> inverse-distance weighted mean of top-3
    scores.sort(key=lambda x: x[0])
    top = scores[:3]
    w = [1.0 / (d + 1e-6) for d, _ in top]
    return float(sum(wi * dy for wi, (_, dy) in zip(w, top)) / sum(w))


def est_knn(q, pool, wb=0.45, ws=0.30, wyi=0.25):
    ref = [p for p in pool if p["cond"] == q["cond"]] or pool

    def rng(vals):
        mn, mx = min(vals), max(vals)
        return mn, (mx - mn) if mx != mn else 1.0

    bm, br = rng([p["b"] for p in ref]); sm, sr = rng([p["s"] for p in ref]); ym, yr = rng([p["yi"] for p in ref])
    wt = wb + ws + wyi
    wb, ws, wyi = wb / wt, ws / wt, wyi / wt
    tb, ts, ty = (q["b"] - bm) / br, (q["s"] - sm) / sr, (q["yi"] - ym) / yr
    sc = []
    for p in ref:
        d = (wb * (tb - (p["b"] - bm) / br) ** 2 + ws * (ts - (p["s"] - sm) / sr) ** 2
             + wyi * (ty - (p["yi"] - ym) / yr) ** 2) ** 0.5
        sc.append((d, p["day"]))
    return top3_est(sc)


def est_wass(q, pool):
    sc = []
    for p in pool:
        d = float(np.sum(np.abs(np.cumsum(q["hist"]) - np.cumsum(p["hist"])))) / len(q["hist"])
        sc.append((d, p["day"]))
    return top3_est(sc)


def est_fft(q, pool):
    hfv = [p["fft"][0] for p in pool]; env = [p["fft"][1] for p in pool]
    hf_r = max(max(hfv) - min(hfv), 1e-6); ent_r = max(max(env) - min(env), 1e-6)
    sc = []
    for p in pool:
        d_hf = abs(q["fft"][0] - p["fft"][0]) / hf_r
        d_ent = abs(q["fft"][1] - p["fft"][1]) / ent_r
        dot = np.dot(q["fft"][2], p["fft"][2])
        cos_d = 1.0 - dot / (np.linalg.norm(q["fft"][2]) * np.linalg.norm(p["fft"][2]) + 1e-12)
        sc.append((0.5 * d_hf + 0.3 * d_ent + 0.2 * cos_d, p["day"]))
    return top3_est(sc)


def est_spatial(q, pool):
    env = [p["sp"][0] for p in pool]; bgv = [p["sp"][1] for p in pool]
    ent_r = max(max(env) - min(env), 1e-6)
    bg_r = max(max(abs(v) for v in bgv) * 2, 1e-6)
    sc = []
    for p in pool:
        d = (0.4 * abs(q["sp"][0] - p["sp"][0]) / ent_r + 0.4 * abs(q["sp"][1] - p["sp"][1]) / bg_r
             + 0.2 * abs(q["sp"][2] - p["sp"][2]) / (abs(p["sp"][2]) + 1))
        sc.append((d, p["day"]))
    return top3_est(sc)


# --------------------------------------------------------- Huber ensemble weight
M4 = ["knn", "wass", "fft", "spatial"]


def opt_w(train):  # train = [(true_day, {method: estimate})]
    A = np.array([[tr[1][m] for m in M4] for tr in train], float)
    y = np.array([tr[0] for tr in train], float)

    def loss(w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            return 1e12
        e = A @ (w / s) - y
        ae = np.abs(e)
        return float(np.mean(np.where(ae <= HUBER_DELTA, 0.5 * e * e,
                                      HUBER_DELTA * (ae - 0.5 * HUBER_DELTA))))

    bw = np.ones(4) / 4
    bl = loss(bw)
    starts = [np.ones(4) / 4] + [np.where(np.arange(4) == i, 0.80, 0.05) for i in range(4)]
    for x0 in starts:
        r = minimize(loss, x0, bounds=[(0, 1)] * 4, method="L-BFGS-B")
        if r.fun < bl:
            bl, bw = r.fun, np.maximum(r.x, 0.0)
    return bw / bw.sum()


def rmse(e):
    e = np.asarray([x for x in e if np.isfinite(x)], float)
    return float(np.sqrt(np.mean(e * e))) if len(e) else float("nan")


# --------------------------------------------- A1g 4-OLS R^2-weighted LOO (20 p)
def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm) ** 2).sum()
    if Sxx == 0:
        return ym, 0.0, 0.0
    beta = ((x - xm) * (y - ym)).sum() / Sxx
    alpha = ym - beta * xm
    yh = alpha + beta * x
    sst = ((y - ym) ** 2).sum()
    r2 = 1 - ((y - yh) ** 2).sum() / sst if sst > 0 else 0.0
    return alpha, beta, max(0.0, r2)


def a1g_loo(pts):
    """pts = [(b, S, YI, dE, a1g)] -> (R2, RMSE, r, MAE)"""
    refs, preds = [], []
    for i in range(len(pts)):
        tr = pts[:i] + pts[i + 1:]
        y = [p[4] for p in tr]
        num = den = 0.0
        for k in range(4):
            a, b, r2 = ols([p[k] for p in tr], y)
            num += r2 * (a + b * pts[i][k])
            den += r2
        preds.append(num / den if den > 0 else float(np.mean(y)))
        refs.append(pts[i][4])
    refs = np.array(refs); preds = np.array(preds)
    err = preds - refs
    ss_res = float((err ** 2).sum()); ss_tot = float(((refs - refs.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot, float(np.sqrt((err ** 2).mean())),
            float(np.corrcoef(refs, preds)[0, 1]), float(np.abs(err).mean()))


# ------------------------------------------------------------------------- main
def main():
    dbdir = find_dbdir()
    print("=" * 78)
    print("MSSP-D-26-02329 canonical reproduction — manual-ROI LODO + A1g regression")
    print(f"DB directory: {dbdir}  (read-only)")
    print("=" * 78)

    # ---- 1. data structure --------------------------------------------------
    print("\n## 1. Data structure (hfs2_oxidation_dataset.db)")
    imgs = load_images(dbdir / DB_NAME)
    png = sorted([d for d in imgs if d["name"].lower().endswith(".png")], key=lambda x: x["name"])
    jpg = [d for d in imgs if d["name"].lower().endswith(".jpg")]
    check_int("PNG query images", len(png), 33)
    check_int("JPG reference images", len(jpg), 20)
    check_int("total images", len(imgs), 53)
    jpg_keys = {(p["cond"], p["day"]) for p in jpg}
    overlap = sum(1 for q in png if (q["cond"], q["day"]) in jpg_keys)
    check_int("exact (condition, day) overlaps PNG-JPG", overlap, 16)

    # ---- 1b. PIXEL-FIDELITY GATE -------------------------------------------
    # Recompute the stored colour descriptors from the raw rgb_blob + saved ROI and
    # require them to equal the stored values. This makes every downstream headline
    # metric a function of the pixels: corrupt an image and this gate fails first.
    print("\n## 1b. Pixel-fidelity gate (stored L*/a*/b*/S/YI == recompute from rgb_blob + ROI)")
    maxd = {"L": 0.0, "a": 0.0, "b*": 0.0, "S": 0.0, "YI": 0.0}
    for d in imgs:
        m = roi_mask(d["rgb"].shape, d["roi"])
        L, a, b = recompute_lab(d["rgb"], m)
        maxd["L"] = max(maxd["L"], abs(L - d["L"]))
        maxd["a"] = max(maxd["a"], abs(a - d["a"]))
        maxd["b*"] = max(maxd["b*"], abs(b - d["b"]))
        maxd["S"] = max(maxd["S"], abs(recompute_s_mean(d["rgb"], m) - d["s"]))
        maxd["YI"] = max(maxd["YI"], abs(recompute_yi(d["rgb"], m) - d["yi"]))
    check_true("stored descriptors reproduce from pixels (all 53, max|diff| < 1e-3)",
               max(maxd.values()) < 1e-3,
               "max|diff| " + ", ".join(f"{k}:{v:.2g}" for k, v in maxd.items()))

    # ---- 2. descriptor facts (stored manual-ROI descriptors, 33 PNG) --------
    print("\n## 2. Descriptor facts (33 PNG, stored manual-ROI descriptors)")
    n70 = {d["day"]: d for d in png if d["cond"] == "NativeHfS2-70%RH"}
    check("N70 b* at day 0", n70[0.0]["b"], 33.3, fmt=".1f")
    check("N70 b* at day 14", n70[14.0]["b"], 3.0, fmt=".1f")

    B = np.array([d["b"] for d in png]); S = np.array([d["s"] for d in png])
    YI = np.array([d["yi"] for d in png])
    r_bs = float(np.corrcoef(B, S)[0, 1])
    r_byi = float(np.corrcoef(B, YI)[0, 1])
    r_syi = float(np.corrcoef(S, YI)[0, 1])
    check_true("r(b*,S), r(b*,YI), r(S,YI) all >= 0.997",
               min(r_bs, r_byi, r_syi) >= 0.997,
               f"r={r_bs:.4f}/{r_byi:.4f}/{r_syi:.4f}")

    # ---- 2b. SI Table S1: per-condition r(day, metric) + Fisher-z 95% CI -----
    # The SI states "this script regenerates the correlation coefficients reported
    # here", so we compare 1:1 with the published values. Requiring an exact match
    # after rounding (2dp) catches transcription errors — this check actually found
    # two (N70 b* CI lower bound, PMMA YI CI upper bound).
    print("\n## 2b. SI Table S1 — r(day, metric) with Fisher-z 95% CI (33 PNG)")
    TABLE_S1 = {  # cond: {metric: (r, ci_low, ci_high)} — SI Table S1 published values
        "NativeHfS2-35%RH": {"b*": (-0.95, -1.00, -0.39), "S": (-0.95, -1.00, -0.43),
                             "YI": (-0.98, -1.00, -0.71)},
        "NativeHfS2-70%RH": {"b*": (-0.81, -0.95, -0.40), "S": (-0.80, -0.95, -0.39),
                             "YI": (-0.82, -0.95, -0.43)},
        "Al2O3HfS2-70%RH": {"b*": (-0.46, -0.82, +0.15), "S": (-0.44, -0.81, +0.17),
                            "YI": (-0.45, -0.82, +0.16)},
        "PMMA HfS2-70%RH": {"b*": (-0.86, -0.99, +0.10), "S": (-0.82, -0.99, +0.23),
                            "YI": (-0.84, -0.99, +0.17)},
    }

    def fisher_ci(r, n):
        """Fisher-z 95% CI. Undefined for n<4 or |r|=1, in which case return [-1,1]."""
        if n < 4 or abs(r) >= 1.0:
            return -1.0, 1.0
        z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
        return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))

    for cond in CONDS:
        grp = [d for d in png if d["cond"] == cond]
        dy = np.array([d["day"] for d in grp], float)
        for metric, key in (("b*", "b"), ("S", "s"), ("YI", "yi")):
            vals = np.array([d[key] for d in grp], float)
            r = float(np.corrcoef(dy, vals)[0, 1])
            lo, hi = fisher_ci(r, len(grp))
            want_r, want_lo, want_hi = TABLE_S1[cond][metric]
            got = (round(r, 2), round(lo, 2), round(hi, 2))
            check_true(f"Table S1 {cond} r(day,{metric})",
                       got == (want_r, want_lo, want_hi),
                       f"n={len(grp)} got {got[0]:+.2f} [{got[1]:+.2f}, {got[2]:+.2f}] "
                       f"vs SI {want_r:+.2f} [{want_lo:+.2f}, {want_hi:+.2f}]")

    # ---- 3. dE is same-pool and reproducible from Lab alone -----------------
    # Reference for each condition = that condition's PNG day-0 Lab (same-pool).
    # The stored delta_e must equal this recomputation and must be 0 at day 0.
    print("\n## 3. delta_e is same-pool (reference = PNG day-0 Lab of same condition)")
    ref0 = {d["cond"]: d for d in png if d["day"] == 0}
    check_true("one PNG day-0 reference per condition", len(ref0) == 4, f"{len(ref0)}/4 conditions")
    for d in png:
        r = ref0[d["cond"]]
        d["de_new"] = float(np.sqrt((d["L"] - r["L"]) ** 2 + (d["a"] - r["a"]) ** 2
                                    + (d["b"] - r["b"]) ** 2))
    stored_d0 = {c: ref0[c]["de"] for c in ref0}
    check_true("stored PNG day-0 delta_e == 0 (same-pool reference is day-0 of same condition)",
               all(abs(v) < 1e-6 for v in stored_d0.values()),
               "stored day-0 dE = " + ", ".join(f"{c}:{v:.4f}" for c, v in sorted(stored_d0.items())))
    max_de_diff = max(abs(d["de"] - d["de_new"]) for d in png)
    check_true("stored PNG delta_e equals same-pool recomputation (all 33, tol 1e-4)",
               max_de_diff < 1e-4, f"max|stored - recomputed| = {max_de_diff:.2e}")
    check_true("recomputed dE at day 0 equals 0 for all conditions",
               all(ref0[c]["de_new"] == 0.0 for c in ref0))

    # JPG delta_e is same-pool within the JPG pool (reference = JPG day-0 of same
    # condition). This is the dE that feeds the A1g regression (§8), so anchoring
    # it to the pixel-verified Lab (§1b) closes the last "ledger" gap: corrupt a
    # JPG's stored dE and this check fails.
    jref0 = {d["cond"]: d for d in jpg if d["day"] == 0}
    check_true("one JPG day-0 reference per condition", len(jref0) == 4, f"{len(jref0)}/4 conditions")
    for d in jpg:
        r = jref0[d["cond"]]
        d["de_new"] = float(np.sqrt((d["L"] - r["L"]) ** 2 + (d["a"] - r["a"]) ** 2
                                    + (d["b"] - r["b"]) ** 2))
    max_jde = max(abs(d["de"] - d["de_new"]) for d in jpg)
    check_true("stored JPG delta_e equals same-pool recomputation (all 20, tol 1e-4)",
               max_jde < 1e-4, f"max|stored - recomputed| = {max_jde:.2e}")

    # ---- 4. LODO (manual ROI): features, estimators, ensemble, baselines ----
    print("\n## 4. LODO, manual (saved) ROI — computing features on 53 images ...")
    for d in imgs:
        m = roi_mask(d["rgb"].shape, d["roi"])
        d["hist"] = hist_sig(d["rgb"], m)
        d["fft"] = fft_feat(d["rgb"], m)
        d["sp"] = spatial_feat(d["rgb"], m, d["roi"])
    jpg_ox = [d for d in jpg if d["cond"] in OX]
    check_int("oxidizing-condition JPG references", len(jpg_ox), 15)

    R = []
    for q in png:
        pool = [p for p in jpg if not (p["cond"] == q["cond"] and p["day"] == q["day"])]
        est = {"knn": est_knn(q, pool), "wass": est_wass(q, pool),
               "fft": est_fft(q, pool), "spatial": est_spatial(q, pool)}
        tr_ox = [p for p in jpg_ox if not (p["cond"] == q["cond"] and p["day"] == q["day"])]
        pm = float(np.mean([p["day"] for p in pool]))
        bs = bs_clip = np.nan
        if q["cond"] in OX:
            X = np.array([p["b"] for p in tr_ox]); y = np.array([p["day"] for p in tr_ox])
            A = np.vstack([X, np.ones(len(X))]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            bs = float(coef[0] * q["b"] + coef[1])
            bs_clip = float(np.clip(bs, CLIP_LO, CLIP_HI))
        R.append(dict(name=q["name"], cond=q["cond"], day=q["day"], pool=len(pool),
                      est=est, pm=pm, bs=bs, bs_clip=bs_clip))

    n_excl = sum(1 for r in R if r["pool"] == 19)
    check_int("LODO folds with an exclusion (= overlaps)", n_excl, 16)
    ox_q = [r for r in R if r["cond"] in OX]
    check_int("oxidizing-condition queries (headline n)", len(ox_q), 21)

    # ensemble: fold-wise Huber refit on the remaining 20 ox queries
    plain = [(r["day"], r["est"]) for r in ox_q]
    for i, r in enumerate(ox_q):
        w = opt_w([plain[k] for k in range(21) if k != i])
        r["ens"] = float(np.dot(w, [r["est"][m] for m in M4]))
    w_all = opt_w(plain)

    def errs(rows, key):
        if key in ("ens", "pm", "bs", "bs_clip"):
            v = np.array([r[key] for r in rows], float)
        else:
            v = np.array([r["est"][key] for r in rows], float)
        return v - np.array([r["day"] for r in rows], float)

    print("\n## 5. Headline RMSE (oxidizing 3 conditions, n = 21, days)")
    for key, label, want in [("bs_clip", "b* regression + clip[0,29]", 6.78),
                             ("knn", "kNN color distance", 7.70),
                             ("ens", "4-method Huber ensemble", 7.94),
                             ("spatial", "spatial", 9.95),
                             ("wass", "Wasserstein", 8.77),
                             ("fft", "FFT", 12.78),
                             ("pm", "pool-mean (LODO baseline)", 9.84)]:
        check(f"RMSE {label}", rmse(errs(ox_q, key)), want)

    print("\n## 6. Ensemble weights (Huber fit, ox-21)")
    for m, want in zip(M4, [0.91, 0.07, 0.00, 0.02]):
        check(f"ensemble weight {m}", float(w_all[M4.index(m)]), want)

    # ---- 7. significance ----------------------------------------------------
    print("\n## 7. Significance (paired bootstrap B=20000 seed=0 + Wilcoxon, n=21)")

    def paired(ka, kb):
        ea, eb = errs(ox_q, ka), errs(ox_q, kb)
        rng_ = np.random.default_rng(BOOT_SEED)
        idx = rng_.integers(0, len(ea), (BOOT_B, len(ea)))
        dd = np.sqrt(np.mean(eb[idx] ** 2, axis=1)) - np.sqrt(np.mean(ea[idx] ** 2, axis=1))
        lo, hi = np.percentile(dd, [2.5, 97.5])
        aa, ab = np.abs(ea), np.abs(eb)
        p2 = float(wilcoxon(aa, ab).pvalue)
        wins = int(np.sum(aa < ab))
        return rmse(ea), rmse(eb), float(lo), float(hi), p2, wins

    ra, rb, lo, hi, p2, wins = paired("bs_clip", "pm")
    check("b*+clip vs pool-mean dRMSE", rb - ra, 3.06)
    check("b*+clip vs pool-mean bootstrap CI low", lo, 1.55)
    check("b*+clip vs pool-mean bootstrap CI high", hi, 4.42)
    check("b*+clip vs pool-mean Wilcoxon two-sided p", p2, 0.0030, tol=0.0005, fmt=".4f")
    check_int("b*+clip vs pool-mean wins (|err| smaller)", wins, 18)
    check_true("b*+clip vs pool-mean CI excludes 0 (significant)", lo > 0.0,
               f"CI [{lo:+.2f}, {hi:+.2f}]")

    ra, rb, lo, hi, p2, wins = paired("knn", "pm")
    check("kNN vs pool-mean bootstrap CI low", lo, 0.17)
    check("kNN vs pool-mean bootstrap CI high", hi, 4.78)
    check_true("kNN vs pool-mean CI excludes 0 (significant)", lo > 0.0,
               f"CI [{lo:+.2f}, {hi:+.2f}], Wilcoxon two-sided p={p2:.4f}")

    # ---- 8. A1g regression (stored dE, 20 Raman-linked JPG pairs) -----------
    print("\n## 8. A1g 4-OLS R^2-weighted LOO (20 Raman-linked JPG pairs, stored dE)")
    a1g_map = {(c, float(d)): MEASURED_A1G[c][i] for c in CONDS for i, d in enumerate(RAMAN_DAYS)}
    exact = [d for d in jpg if (d["cond"], d["day"]) in a1g_map]
    check_int("Raman-linked JPG (condition, day) pairs", len(exact), 20)
    for c in CONDS:
        check_int(f"  {c} exact JPG-Raman pairs", sum(1 for d in exact if d["cond"] == c), 5)
    pts = [(d["b"], d["s"], d["yi"], d["de"], a1g_map[(d["cond"], d["day"])]) for d in exact]
    r2, rm_, r_, mae = a1g_loo(pts)
    check("A1g LOO R^2", r2, 0.600, fmt=".3f")
    check("A1g LOO RMSE", rm_, 0.199, fmt=".3f")
    check("A1g LOO r", r_, 0.794, fmt=".3f")
    check("A1g LOO MAE", mae, 0.159, fmt=".3f")

    # ---- 9. Table 1 decay from Ref.[13] values ------------------------------
    print("\n## 9. Table 1 A1g decay percentages (Ref.[13] values)")
    for c, want in zip(CONDS, [60.1, 94.0, 13.5, 47.3]):
        check(f"decay % {c}", (1 - MEASURED_A1G[c][-1]) * 100, want, fmt=".1f")

    # ---- summary ------------------------------------------------------------
    n_pass = sum(1 for _, ok in CHECKS if ok)
    n_fail = len(CHECKS) - n_pass
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{len(CHECKS)} checks PASS, {n_fail} FAIL")
    if n_fail:
        for name, ok in CHECKS:
            if not ok:
                print(f"  FAILED: {name}")
        print("RESULT: FAIL — canonical values NOT fully reproduced")
        sys.exit(1)
    print("RESULT: ALL PASS — canonical paper values reproduced from the public DB alone")


if __name__ == "__main__":
    main()
