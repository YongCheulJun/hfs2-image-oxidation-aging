# Hyperparameters

Every hyperparameter used to produce the reported numbers, with its value and the
exact source line. All are fixed constants in the code — none are tuned on the test
data, and `reproduce_canonical.py` recomputes the reported values from them (62/62).

## Evaluation protocol

| Parameter | Value | Source |
| --- | --- | --- |
| Reference set | 20 Raman-linked JPG | `reproduce_canonical.py` §2 |
| Query set | 33 analysis PNG | `reproduce_canonical.py` §2 |
| Leave-out rule (LODO) | remove every JPG sharing the query's exact (condition, day) | `reproduce_canonical.py` §2 |
| Headline pool | 3 oxidizing conditions, n = 21 | `OX` — `reproduce_canonical.py:80`, `lodo_lib.py:18` |
| Oxidizing conditions | NativeHfS2-35%RH, NativeHfS2-70%RH, PMMA HfS2-70%RH | `OX` |

`Al2O3HfS2-70%RH` (the passivated condition) is excluded from the headline pool.

## Primary estimator — b\* regression + clip

| Parameter | Value | Source |
| --- | --- | --- |
| Model | OLS of day on b\* over the fold's oxidizing-pool | `reproduce_canonical.py` §5 |
| Prediction clip | [0, 29] days (`CLIP_LO`, `CLIP_HI`) | `reproduce_canonical.py:84`, applied `:532` |

On this data only the lower clip bound (0) ever activates.

## kNN colour-distance estimator

| Parameter | Value | Source |
| --- | --- | --- |
| Feature weights (b\*, S, YI) | 0.45 / 0.30 / 0.25 (renormalized) | `est_knn` — `lodo_lib.py:103`, `reproduce_canonical.py:264` |
| Neighbours | top-3, inverse-distance weighted mean | `top3_est` — `lodo_lib.py:96`, `reproduce_canonical.py:257` |
| Per-condition range normalization | min–max within the condition (fallback: whole pool) | `est_knn` |

## FFT estimator

| Parameter | Value | Source |
| --- | --- | --- |
| High-frequency ring | radius > 0.40 · max radius | `lodo_lib.py:58`, `reproduce_canonical.py:214` |
| Total-power ring | radius > 0.02 · max radius | `lodo_lib.py:59`, `reproduce_canonical.py:215` |
| Radial-profile bins | 64 | `fft_feat` |
| Distance weights (hf / entropy / cosine) | 0.5 / 0.3 / 0.2 | `lodo_lib.py:133`, `reproduce_canonical.py:300` |

## Spatial estimator

| Parameter | Value | Source |
| --- | --- | --- |
| Grid | 3 × 3 | `spatial_feat` — `lodo_lib.py:70`, `reproduce_canonical.py:228` |
| Min pixels per cell | 5 | `spatial_feat` |
| Descriptor | inter-cell b\* std, centre–boundary b\* gradient, anisotropy | `spatial_feat` |
| Distance weights (inter-cell std / centre-boundary gradient / anisotropy) | 0.4 / 0.4 / 0.2 | `lodo_lib.py`, `reproduce_canonical.py` |

The centre–boundary gradient uses the single central cell as the centre and the
surrounding 8 cells as the boundary (mean b\* over the cells with enough pixels).

## Wasserstein estimator

| Parameter | Value | Source |
| --- | --- | --- |
| Histogram | 64 bins of b\* over [-30, 80] | `hist_sig` — `lodo_lib.py:43` |
| Distance | 1-D Wasserstein (cumulative-histogram L1) | `est_wass` — `lodo_lib.py:117` |

## Huber ensemble (supplementary)

| Parameter | Value | Source |
| --- | --- | --- |
| Members (order) | kNN, Wasserstein, FFT, spatial (`M4`) | `lodo_lib.py:148`, `reproduce_canonical.py` §6 |
| Huber delta | 5.0 days | `HUBER_D`/`HUBER_DELTA` — `lodo_lib.py:19`, `reproduce_canonical.py:83` |
| Weight bounds | [0, 1] each, renormalized to sum 1 | `opt_w` — `lodo_lib.py:149`, `reproduce_canonical.py:320` |
| Optimizer | L-BFGS-B | `opt_w` |
| Multistart | uniform + 4 one-hot starts (0.80 / 0.05) | `lodo_lib.py:158`, `reproduce_canonical.py:336` |
| Fit protocol | per-fold refit on the other 20 oxidizing queries | `reproduce_canonical.py` §6 |
| Fitted weights | kNN 0.91 / Wasserstein 0.05 / FFT 0.00 / spatial 0.04 | asserted `reproduce_canonical.py` §6 |

## Significance test

| Parameter | Value | Source |
| --- | --- | --- |
| Bootstrap resamples | 20000 (`BOOT_B`) | `reproduce_canonical.py:85` |
| Bootstrap seed | 0 (`BOOT_SEED`) | `reproduce_canonical.py:85`, used `:575` |
| Paired test | Wilcoxon two-sided | `reproduce_canonical.py` §7 |

## A1g Raman calibration

| Parameter | Value | Source |
| --- | --- | --- |
| A1g values | Ref. [13] published, per (condition, day) | `dataset/raman_a1g_values.csv` |
| Pairs | 20 Raman-linked JPG | `reproduce_canonical.py` §8 |
| Model | 4-OLS (b\*, S, YI, ΔE), R²-weighted, leave-one-out | `a1g_loo` — `reproduce_canonical.py` §8 |
| ΔE reference | same-pool (JPG day-0 of the same condition) | `reproduce_canonical.py` §8 |

## Tolerance

| Parameter | Value | Source |
| --- | --- | --- |
| Default check tolerance | ±0.05 (integer checks: exact; p-value: ±0.0005) | `check` — `reproduce_canonical.py` |
