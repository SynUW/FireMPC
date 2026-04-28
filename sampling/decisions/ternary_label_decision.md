# Ternary Label Design Decision

**Date:** 2026-04-04
**Status:** FINAL — Phase 1 calibrated decision, implementation binding for Phase 2
**Decision ID:** D-LABEL-01
**Requirement:** DATA-05

---

## Decision

MODIS MCD14A1 firemask values are mapped to three label classes:

| Firemask Value | Class | Use in Dataset |
|----------------|-------|----------------|
| firemask == 9  | High-confidence fire | **Positive training label** |
| firemask == 8  | Nominal-confidence fire | Exclusion buffer only — NOT a positive label, NOT a negative candidate |
| firemask == 7  | Low-confidence fire | Exclusion buffer only — NOT a positive label, NOT a negative candidate |
| firemask == 4  | Cloud-obscured | Excluded from negative candidates |
| firemask == 5  | Non-fire land | Valid negative candidate (subject to Layer 1 buffer radius) |
| firemask == 3  | Water | Excluded from all sampling |
| firemask == 0  | Not processed | Excluded from all sampling |

**Effective label scheme:** ternary in the exclusion sense (positive / uncertain-excluded / negative-candidate), but binary in training labels (positive = 1, confirmed negative = 0).

---

## Rationale

**Why exclude firemask == 8 (nominal confidence) from positive labels?**

1. **Reviewer credibility:** MODIS active fire detection at nominal confidence (~68% precision in dense vegetation; Giglio et al. 2020) introduces substantial false positives. Using only high-confidence detections (firemask == 9, >80% precision) maximizes label reliability — a defensible choice for a NeurIPS benchmark dataset.

2. **Conservative sampling:** Including nominal-confidence pixels as positive labels would inflate the positive count but reduce label quality. For a benchmark, label quality matters more than count.

3. **Exclusion buffer preserves safety margin:** By including firemask == 8 and firemask == 7 in the exclusion buffer (not positive, not negative), we prevent nominal-confidence fire pixels from contaminating the negative candidate pool. This is the most conservative approach.

4. **Standard practice alignment:** The Fire Information for Resource Management System (FIRMS) distinguishes high-confidence from nominal-confidence. Our choice follows the same conservative convention.

---

## Impact on Training Set Size

The following statistics are from the full MODIS scan (scope_audit_report.json):

- **High-confidence positives (firemask == 9):** 2,176,363 pixel-days
- **Nominal-confidence (firemask == 8, excluded from labels):** 2,430,234 pixel-days
- **Low-confidence (firemask == 7, excluded from labels):** 605,895 pixel-days
- **Ratio of excluded-but-buffered to positive labels:** 1.12x (nominal-to-positive)

If medium-confidence (firemask == 8) were included as positive labels, the positive count would increase by approximately 112%. This is noted as a sensitivity check but NOT recommended for v1.

**Dataset span:** 2000–2024 based on available tiles.

---

## Implementation Notes

- `audit.modis_utils.get_positive_mask(firemask)` returns `firemask == 9` only
- `audit.modis_utils.get_exclusion_mask(firemask)` returns `firemask in {4, 7, 8, 9}`
- Phase 2 Layer 1 reliability filter applies spatial/temporal buffer around ALL firemask in {7, 8, 9}
- Medium-confidence pixels (firemask == 8) do NOT appear in training labels or negative candidates

---

## Alternative Considered

**Including firemask == 8 as positive labels:** Would increase positive count by 112% but reduce precision below the high-confidence threshold. Rejected for v1; could be explored as a v2 sensitivity analysis (vary confidence threshold).
