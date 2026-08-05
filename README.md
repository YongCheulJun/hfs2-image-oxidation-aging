# HfS₂ Image-Based Oxidation-Aging — Data & Reproduction

Data and code for reproducing the aging-day and Raman-A₁g calibration results of

> **Raman-calibrated image-based screening of ambient aging in CVD-grown HfS₂ thin films** (MSSP-D-26-02329)
> Yongcheul Jun, Juchan Hwang, Tan Chee Leong, Kwangwook Park (corresponding author, kwangwook.park@jbnu.ac.kr)

```bash
pip install -r requirements.txt
python tools/reproduce_canonical.py     # recomputes & asserts the reported values → 139/139 PASS
python tools/reproduce_applicability.py # reproduces the operating-window section (tau, tsat, gate) → 12/12 PASS
python tools/make_foldwise_csv.py       # writes outputs/canon_foldwise.csv
```

Python ≥ 3.10 with numpy, scipy, opencv-python, Pillow. The SQLite database is
opened read-only and is never modified. `reproduce_canonical.py` only recomputes
and asserts, writing nothing; `make_foldwise_csv.py` regenerates the output file
`outputs/canon_foldwise.csv` (identical to the committed copy).

## Contents

| Path | Description |
| --- | --- |
| `dbfiles/hfs2_oxidation_dataset.db` | 53 images (33 PNG queries + 20 JPG reference library): RGB, ROI, descriptors (L\*, a\*, b\*, S, YI, ΔE). The aging-day (LODO) reproduction reads its image data from this file; the A₁g regression additionally reads `dataset/raman_a1g_values.csv`. |
| `dbfiles/manifest.csv` | Plain index of the 53 images (condition, day, ROI, descriptors). |
| `dbfiles/CHECKSUMS.md5` | MD5 checksums of the data files. |
| `dataset/images/` | 33 analysis photographs (PNG) — aging-day queries. |
| `dataset/images_raman_jpg/` | 20 Raman-linked photographs (JPG) — reference library. |
| `dataset/raman_a1g_values.csv` | Ref. [13] published normalized A₁g per (condition, day) — the single source read by the A₁g regression (nothing hardcoded). |
| `tools/reproduce_canonical.py` | Recomputes and asserts the reported values (139 checks: data structure, pixel-fidelity, descriptors, pooled + per-condition LODO RMSE, weights, significance, pooled A₁g + Table S5 OLS/LOPO incl. the ΔE-only headline, Table 2 per-condition agreement + decay fits, Table 1 decay %, ROI-rescaling sensitivity bound) against the DB. |
| `tools/reproduce_applicability.py` | Reproduces the "Applicability criterion and operating window" section from the deposited b\* data (12 checks: per-condition offset-exponential fit τ, saturation time t_sat, local day resolution, operating-window gate counts). |
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
| per-condition RMSE (b\* / kNN / pool-mean) | 7.87·4.37·9.44 / 9.64·5.12·9.92 / 10.50·9.20·10.50 d (N35 / N70 / PMMA); Al₂O₃ control kNN 9.17 vs 9.48 d |
| Raman A₁g calibration (n=20) | R² = 0.600; best single descriptor ΔE R² = 0.701 (Table S5) |
| Table 2 per-condition A₁g | r 0.96 / 0.88 / 0.41 / 0.81; decay k 0.036 / 0.262 / 0.005 / 0.029 day⁻¹ |
| ROI-rescaling sensitivity | b\* LODO RMSE changes ≤ 0.03 d over −40%..0% area rescale |
| Operating window (b\* offset-exp fit) | τ = 12.6 / 3.36 / 5.06 d; gate flags 9 (2σ=1.4) or 12 (2σ=4.1) of 21 |

See [`REPRODUCE.md`](REPRODUCE.md) for the number-to-command mapping.

## Contact

- Kwangwook Park (corresponding author) — kwangwook.park@jbnu.ac.kr
