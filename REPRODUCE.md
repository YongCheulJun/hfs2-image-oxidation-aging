# Reproducing the reported numbers

```bash
python3 -m pip install -r requirements.txt     # numpy, scipy, opencv-python, Pillow
python3 tools/reproduce_canonical.py           # 62/62 PASS
python3 tools/make_foldwise_csv.py             # outputs/canon_foldwise.csv
```

`reproduce_canonical.py` reads `dbfiles/hfs2_oxidation_dataset.db` (read-only)
and the deposited Ref. [13] A₁g values `dataset/raman_a1g_values.csv`, imports no
application code, and asserts each reported value
against what it recomputes. Deterministic — same input, same output.

Verify data integrity: `md5sum -c dbfiles/CHECKSUMS.md5`

**Evaluation protocol.** The metric is a query-excluded leave-one-day-out: the
reference set is the 20 JPG, and for each of the 33 PNG queries every JPG sharing
the query's exact (condition, day) is removed before estimation. The 33 PNG and
20 JPG include 16 same-(condition, day) pairs, so a plain leave-one-out over all
53 records is **not** the protocol used here and would understate the error.

## Number → command mapping

| Value | Source |
|-------|--------|
| 6.78 d (headline, b\*+clip) | `reproduce_canonical.py` §5 |
| 7.70 d (kNN) | `reproduce_canonical.py` §5 |
| 7.94 d (4-method ensemble) | `reproduce_canonical.py` §5 |
| 9.84 d (pool-mean baseline) | `reproduce_canonical.py` §5 |
| ensemble weights (kNN 0.91 / Wasserstein 0.05 / spatial 0.04 / FFT 0.00) | `reproduce_canonical.py` §6 |
| significance (bootstrap CI, Wilcoxon, wins) | `reproduce_canonical.py` §7 |
| Raman A₁g R² 0.600 / RMSE 0.199 / r 0.794 / MAE 0.159 | `reproduce_canonical.py` §8 (A₁g from `dataset/raman_a1g_values.csv`) |
| Table 1 A₁g decay % (60.1 / 94.0 / 13.5 / 47.3) | `reproduce_canonical.py` §9 |
| SI Table S1 correlation coefficients + CIs | `reproduce_canonical.py` §2b |
| SI fold-wise prediction table | `make_foldwise_csv.py` → `outputs/canon_foldwise.csv` |
| operating window: τ (12.6 / 3.36 / 5.06 d), t_sat (23 / 10 / 8 d), native-70%RH local day resolution, gate flags (9 at 2σ=1.4, 12 at 2σ=4.1, of 21) | `reproduce_applicability.py` §1–4 (b\* offset-exponential fit on the deposited data) |
| the reported values above, asserted (62 + 12 checks) | `reproduce_canonical.py` and `reproduce_applicability.py` |
