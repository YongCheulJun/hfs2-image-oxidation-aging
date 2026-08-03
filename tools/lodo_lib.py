# Function library for the canonical LODO reproduction — descriptor extraction,
# the 4 estimators, and the ensemble weighting, all from the public deposit DB
# (hfs2_oxidation_dataset.db) alone.
"""Imported by `make_foldwise_csv.py`. The DB is always opened `?mode=ro` (never
modified). This is the code path for the canonical values (3 oxidizing conditions,
n=21, leave-one-day-out) — see REPRODUCE.md for details. The evaluation protocol
(removing every JPG sharing the query's (condition, day) in each fold) is exactly
what prevents the twin-record leak.
"""
import os, io, sqlite3
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/dbfiles/hfs2_oxidation_dataset.db"
OX = ["NativeHfS2-35%RH", "NativeHfS2-70%RH", "PMMA HfS2-70%RH"]
HUBER_D = 5.0

def load(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT name, day, cond, roi_x0, roi_y0, roi_x1, roi_y1, "
                       "s_mean, yellowness_idx, lab_L, lab_a, lab_b, delta_e, rgb_blob FROM images").fetchall()
    con.close()
    out = []
    for r in rows:
        rgb = np.array(Image.open(io.BytesIO(r[13])).convert("RGB"))
        out.append(dict(name=r[0], day=float(r[1]), cond=r[2], roi=(r[3], r[4], r[5], r[6]),
                        s=float(r[7]), yi=float(r[8]), L=float(r[9]), a=float(r[10]),
                        b=float(r[11]), de=float(r[12]), rgb=rgb))
    return out

# ---------- Features (self-contained implementation) ----------
def roi_mask(shape, roi):
    m = np.zeros(shape[:2], bool); x0, y0, x1, y1 = roi; m[y0:y1, x0:x1] = True
    return m

def bstar_map(rgb):
    lab = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab).astype(np.float32)
    return lab[:, :, 2] - 128.0

def hist_sig(rgb, m, bins=64):
    b = bstar_map(rgb)[m]
    h, _ = np.histogram(b, bins=bins, range=(-30.0, 80.0))
    t = h.sum()
    return (np.ones(bins) / bins) if t == 0 else h.astype(np.float64) / t

def fft_feat(rgb, m):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mv = float(gray[m].mean())
    g = np.full_like(gray, mv); g[m] = gray[m]
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    H, W = P.shape; cy, cx = H // 2, W // 2
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
        if ring.sum() > 0: radial[i] = float(P[ring].mean())
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
            if len(vals) >= 5: seg[r, c] = float(np.mean(vals))
    ent = float(np.nanstd(seg))
    # Original definition: in the 3x3 grid the centre block spans everything, so the
    # boundary is empty -> grad = 0 (reproduced exactly as defined).
    cr, cc = rows // 2, cols // 2
    cm = np.zeros((rows, cols), bool); cm[max(0, cr - 1):cr + 2, max(0, cc - 1):cc + 2] = True
    bd = ~cm
    cv_ = seg[cm & ~np.isnan(seg)]; bv = seg[bd & ~np.isnan(seg)]
    grad = float(np.mean(bv) - np.mean(cv_)) if (len(cv_) and len(bv)) else 0.0
    rv = [np.nanvar(seg[r, :]) for r in range(rows) if not np.all(np.isnan(seg[r, :]))]
    cvv = [np.nanvar(seg[:, c]) for c in range(cols) if not np.all(np.isnan(seg[:, c]))]
    rvar = float(np.mean(rv)) if rv else 0.0
    cvar = float(np.mean(cvv)) if cvv else 0.0
    ani = rvar / (cvar + 1e-6) if cvar > 1e-6 else 1.0
    return ent, grad, ani

def top3_est(scores):  # [(dist, day)] -> inv-dist weighted mean of top3
    scores.sort(key=lambda x: x[0])
    top = scores[:3]
    w = [1.0 / (d + 1e-6) for d, _ in top]
    return float(sum(wi * dy for wi, (_, dy) in zip(w, top)) / sum(w))

# ---------- Estimators ----------
def est_knn(q, pool, wb=0.45, ws=0.30, wyi=0.25):
    ref = [p for p in pool if p["cond"] == q["cond"]] or pool
    def rng(vals):
        mn, mx = min(vals), max(vals); return mn, (mx - mn) if mx != mn else 1.0
    bm, br = rng([p["b"] for p in ref]); sm, sr = rng([p["s"] for p in ref]); ym, yr = rng([p["yi"] for p in ref])
    wt = wb + ws + wyi; wb, ws, wyi = wb / wt, ws / wt, wyi / wt
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

# ---------- Huber ensemble weighting ----------
M4 = ["knn", "wass", "fft", "spatial"]
def opt_w(train):  # train = [(true_day, {m: est})]
    A = np.array([[tr[1][m] for m in M4] for tr in train], float)
    y = np.array([tr[0] for tr in train], float)
    def loss(w):
        w = np.maximum(w, 0.0); s = w.sum()
        if s <= 0: return 1e12
        e = A @ (w / s) - y; ae = np.abs(e)
        return float(np.mean(np.where(ae <= HUBER_D, 0.5 * e * e, HUBER_D * (ae - 0.5 * HUBER_D))))
    bw = np.ones(4) / 4; bl = loss(bw)
    starts = [np.ones(4) / 4] + [np.where(np.arange(4) == i, 0.80, 0.05) for i in range(4)]
    for x0 in starts:
        r = minimize(loss, x0, bounds=[(0, 1)] * 4, method="L-BFGS-B")
        if r.fun < bl: bl, bw = r.fun, np.maximum(r.x, 0.0)
    return bw / bw.sum()

def rmse(e):
    e = np.asarray([x for x in e if np.isfinite(x)], float)
    return float(np.sqrt(np.mean(e * e))) if len(e) else float("nan")

