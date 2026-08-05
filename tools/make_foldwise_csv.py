#!/usr/bin/env python3
# Writes the fold-wise prediction table of the canonical manual-ROI LODO to
# outputs/canon_foldwise.csv (the raw data behind SI Table S8, promised in the Response Letter).
"""
From the public deposit DB (hfs2_oxidation_dataset.db) alone, computes the fold-wise
predictions and errors of the canonical LODO (reference = 20 JPG, query = 33 PNG,
leaving out every JPG sharing the query's (condition, day) in each fold — this is what
prevents the twin-record leak) and writes them to outputs/canon_foldwise.csv. Estimators
and features are reused from tools/lodo_lib.py; auto_detect_roi is not used and the GUI
pipeline (hfs2_v5_49.py) is never imported. The DB is opened ?mode=ro (never modified).
Headline RMSE verification is handled by tools/reproduce_canonical.py; this script only
emits the raw-data CSV.
"""
import os, csv
import numpy as np
import lodo_lib as L

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def feats(d):
    m = L.roi_mask(d["rgb"].shape, d["roi"])
    d["hist"] = L.hist_sig(d["rgb"], m)
    d["fft"] = L.fft_feat(d["rgb"], m)
    d["sp"] = L.spatial_feat(d["rgb"], m, d["roi"])


def main():
    imgs = L.load(L.DB)
    png = sorted([d for d in imgs if d["name"].lower().endswith(".png")], key=lambda x: x["name"])
    jpg = [d for d in imgs if d["name"].lower().endswith(".jpg")]
    assert len(png) == 33 and len(jpg) == 20, f"png={len(png)} jpg={len(jpg)}"
    for d in imgs:
        feats(d)
    jpg_ox = [d for d in jpg if d["cond"] in L.OX]

    R = []
    for q in png:
        pool = [p for p in jpg if not (p["cond"] == q["cond"] and p["day"] == q["day"])]
        excl = [p["name"] for p in jpg if (p["cond"] == q["cond"] and p["day"] == q["day"])]
        est = {"knn": L.est_knn(q, pool), "wass": L.est_wass(q, pool),
               "fft": L.est_fft(q, pool), "spatial": L.est_spatial(q, pool)}
        tr_ox = [p for p in jpg_ox if not (p["cond"] == q["cond"] and p["day"] == q["day"])]
        bs_clip = np.nan
        if q["cond"] in L.OX:
            X = np.array([p["b"] for p in tr_ox]); y = np.array([p["day"] for p in tr_ox])
            A = np.vstack([X, np.ones(len(X))]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            bs_clip = float(np.clip(coef[0] * q["b"] + coef[1], 0.0, 29.0))
        R.append(dict(name=q["name"], cond=q["cond"], day=q["day"], pool=len(pool),
                      excl=excl, est=est, bs_clip=bs_clip))

    # Ensemble: per-fold Huber refit over the 21 oxidizing-condition queries
    ox_q = [r for r in R if r["cond"] in L.OX]
    assert len(ox_q) == 21
    plain = [(r["day"], r["est"]) for r in ox_q]
    for i, r in enumerate(ox_q):
        w = L.opt_w([plain[k] for k in range(21) if k != i])
        r["ens"] = float(np.dot(w, [r["est"][m] for m in L.M4]))

    path = os.path.join(OUT, "canon_foldwise.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_png", "condition", "true_day", "pool_size", "excluded_jpg",
                    "pred_bstar_clip", "err_bstar_clip", "pred_ensemble", "err_ensemble",
                    "pred_knn", "err_knn", "in_main_metric"])
        fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
        for r in R:
            d = r["day"]
            bc, en, kn = r["bs_clip"], r.get("ens", np.nan), r["est"]["knn"]
            w.writerow([r["name"], r["cond"], f"{d:g}", r["pool"], ";".join(r["excl"]),
                        fmt(bc), fmt(bc - d if bc == bc else float("nan")),
                        fmt(en), fmt(en - d if en == en else float("nan")),
                        fmt(kn), fmt(kn - d), "yes" if r["cond"] in L.OX else "no"])
    print(f"[saved] {path}  ({len(R)} rows)")


if __name__ == "__main__":
    main()
