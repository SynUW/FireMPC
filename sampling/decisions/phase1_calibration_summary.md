# Phase 1 Calibration Summary

**Date:** 2026-04-04
**Status:** FINAL — Phase 1 complete. Phase 2 reads from this document.

This document consolidates all Phase 1 calibrated parameters for Phase 2 implementation.
Phase 2 implementers do NOT need to re-read individual Phase 1 output files.

---

## Calibrated Parameters

### Buffer Radius (SAMP-01 — Layer 1 reliability filter)

- **Selected radius:** 30 km
- **Ablated at:** 30, 60, 100 km (see buffer_sensitivity_table.csv for full results)
- **Selection criterion:** Smallest radius retaining adequate negative candidate density (>= 10% of pixels available) across all ecozones
- **Analysis file:** `buffer/outputs/buffer_sensitivity_table.csv`
- **Reference:** Fu et al. (2601.01677) default was 60 km; this analysis updates that value — 30 km yields 100% of ecozones adequate, and is the smallest radius that does so

The `recommended_radius` column in `buffer/outputs/buffer_sensitivity_table.csv` is set to `30` for all ecozone rows.

### FWI Weight Vector (SAMP-03 — Layer 3 hard-negative scoring)

**Weight vector w1–w6 (normalized separation scores):**

| Component | Weight  | Interpretation |
|-----------|---------|----------------|
| FFMC      | 0.209029 | Fine Fuel Moisture Code separation |
| DMC       | 0.151368 | Duff Moisture Code separation |
| DC        | 0.164503 | Drought Code separation |
| ISI       | 0.150533 | Initial Spread Index separation |
| BUI       | 0.161320 | Buildup Index separation |
| FWI       | 0.163247 | Fire Weather Index separation |
| **Sum**   | **1.0**  | Verified (sum_check = 1.0) |

- **Calibration years:** [2017, 2018, 2021]
- **Calibration method:** normalized_separation_score: max(0, (mean_fire - mean_nonfire) / std_nonfire) per component per ecozone, averaged across ecozones; negative separations zeroed
- **File:** `fwi/outputs/weight_vector_w1w6.json`
- **Supporting figure:** `fwi/outputs/fwi_distributions_fire_vs_nonfire.png`

### Dataset Temporal Span (DATA-02)

- **MODIS temporal span:** 2000–2024 (2000-02-24 to 2024-12-31)
- **Total tile count:** 36,160
- **Expected tile count:** 36,312
- **Missing days:** 38 days (see `audit/outputs/scope_audit_report.json` for list)
- **High-confidence positive count:** 2,176,363 pixel-days
- **Total fire days:** 7,706
- **File:** `audit/outputs/scope_audit_report.json`

### Ternary Label Mapping (DATA-05)

| Firemask Value | Label Class | Phase 2 Use |
|----------------|-------------|-------------|
| 9 (high conf)  | Positive label | Training positive samples |
| 8 (nominal conf) | Exclusion buffer | Excluded from all candidates |
| 7 (low conf)   | Exclusion buffer | Excluded from all candidates |
| 4 (cloud)      | Excluded | Excluded from negative candidates |
| 5 (non-fire land) | Negative candidate | Subject to Layer 1 buffer |

Full decision: `decisions/ternary_label_decision.md`

### CLDN/NBAC Status (Phase 1 Plan 6)

- **CLDN:** Out of scope — HTTP 404 on Canada Open Gov portal; no bulk download available (see `decisions/cldn_nbac_gonogo.md`)
- **NBAC:** Out of scope — pre-decided; MODIS MCD14A1 is the sole ground truth source (see STATE.md)

---

## Phase 2 Dependencies

| Phase 2 Plan | Parameter | Value | Source File |
|-------------|-----------|-------|-------------|
| Plan 1 (Layer 1) | Buffer radius | 30 km | `buffer/outputs/buffer_sensitivity_table.csv`, column `recommended_radius` |
| Plan 3 (Layer 3) | Weight vector | See table above | `fwi/outputs/weight_vector_w1w6.json` |
| All plans | Temporal span | 2000–2024 | `audit/outputs/scope_audit_report.json`, keys `modis_start_year`, `modis_end_year` |
| All plans | Label mapping | firemask==9 positive | `decisions/ternary_label_decision.md` |
| All plans | Positive count | 2,176,363 pixel-days | `audit/outputs/scope_audit_report.json`, key `high_conf_positive_count` |

---

## Quick-reference Values for Phase 2 Code

```python
# Copy-paste ready — from Phase 1 calibration outputs
BUFFER_RADIUS_KM = 30
WEIGHT_VECTOR_PATH = "fwi/outputs/weight_vector_w1w6.json"
MODIS_START_YEAR = 2000
MODIS_END_YEAR = 2024
POSITIVE_FIREMASK_VALUE = 9          # firemask == 9
EXCLUSION_FIREMASK_VALUES = {4, 7, 8, 9}
NEGATIVE_CANDIDATE_FIREMASK = 5      # non-fire land
FWI_WEIGHTS = {
    "FFMC": 0.209029,
    "DMC":  0.151368,
    "DC":   0.164503,
    "ISI":  0.150533,
    "BUI":  0.161320,
    "FWI":  0.163247,
}
CALIBRATION_YEARS = [2017, 2018, 2021]
```
