# HfS₂ Image-Based Oxidation-Aging — Data & Reproduction

Data and code for reproducing the aging-day and Raman-A₁g calibration results of

> **Raman-calibrated image-based screening of ambient aging in CVD-grown HfS₂ thin films** (MSSP-D-26-02329)
> Yongcheul Jun, Juchan Hwang, Tan Chee Leong, Kwangwook Park (corresponding author, kwangwook.park@jbnu.ac.kr)

```bash
pip install -r requirements.txt
python tools/reproduce_canonical.py     # recomputes & asserts the reported values → 62/62 PASS
python tools/make_foldwise_csv.py       # writes outputs/canon_foldwise.csv
```

Python ≥ 3.10 with numpy, scipy, opencv-python, Pillow. All DB access is
read-only; nothing is written.

## Contents

| Path | Description |
| --- | --- |
| `dbfiles/hfs2_oxidation_dataset.db` | 53 images (33 PNG queries + 20 JPG reference library): RGB, ROI, descriptors (L\*, a\*, b\*, S, YI, ΔE). The reproduction reads only this file. |
| `dbfiles/manifest.csv` | Plain index of the 53 images (condition, day, ROI, descriptors). |
| `dbfiles/CHECKSUMS.md5` | MD5 checksums of the data files. |
| `dataset/images/` | 33 analysis photographs (PNG) — aging-day queries. |
| `dataset/images_raman_jpg/` | 20 Raman-linked photographs (JPG) — reference library. |
| `dataset/raman_a1g_values.csv` | Ref. [13] published normalized A₁g per (condition, day) — the single source read by the A₁g regression (nothing hardcoded). |
| `tools/reproduce_canonical.py` | Recomputes and asserts the reported values (62 checks: data structure, pixel-fidelity, descriptors, LODO RMSE, weights, significance, A₁g, decay) against the DB. |
| `tools/lodo_lib.py` | Descriptor/estimator library. |
| `tools/make_foldwise_csv.py` | Writes the fold-wise prediction table (the SI fold-wise table). |
|

## Results reproduced

Query-excluded leave-one-day-out (reference = 20 JPG; each JPG sharing the
query's condition+day is removed for that query). Pooled RMSE over the three
oxidizing conditions (n = 21):

| Method | Aging-day RMSE |
| --- | --- |
| b\* regression + clip[0,29] (primary) | 6.78 d |
| kNN colour distance | 7.70 d |
| 4-method Huber ensemble | 7.94 d |
| pool-mean (baseline) | 9.84 d |
| Raman A₁g calibration (n=20) | R² = 0.600 |

See [`REPRODUCE.md`](REPRODUCE.md) for the number-to-command mapping.

## Contact

- Kwangwook Park (corresponding author) — kwangwook.park@jbnu.ac.kr
