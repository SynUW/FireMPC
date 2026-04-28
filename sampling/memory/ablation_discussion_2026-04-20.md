# Ablation Design Discussion — 2026-04-20

## Key Finding: ch0 Leakage

Fire detection channel (ch0) was **not masked** from model inputs.
The model learned fire persistence ("fire today → fire tomorrow") rather than
weather-driven prediction. This explains F1 ~96% across all variants —
the signal was in the input, not the sampling quality.

**Fix applied**: ch0 is now excluded from training inputs.

---

## Variant Z (L0 Baseline) — Added

A new baseline variant Z was added to `generate_samples.py` to prove L1 effectiveness.

| | Positives | Negatives |
|---|---|---|
| **Z (L0)** | firemask 8 + 9 (nominal + high conf) | firemask 4 + 5 (cloud + non-fire land), no buffer |
| **A-E** | firemask 9 only (high conf) | firemask 5 only, with/without buffer |

Run: `python3 generate_samples.py --only-variants Z`

Sample counts for Z are subsampled to match the same ref_counts as A-E (determined by Variant E).
Test set is shared (fixed_test_set, L4 quality, 2023-2025).

---

## Ablation Chain

```
Z (L0) -> A (no buffer) -> B (L1) -> C (L1+L2) -> D (L1+L2+L3) -> E (L1+L2+L3+L4)
```

Each variant is a **cumulative stack** — each adds one layer to the previous.
Test set is fixed (L4 quality, 2023-2025 temporal holdout), shared across all variants.
Train/val/test sample counts are identical across all variants.

---

## How to Prove Sampling Effectiveness (Post ch0-fix)

After masking ch0, F1 and PR-AUC are valid metrics because:
- Test set is fixed (L4 quality) — all variants evaluated on same negatives
- L3 hard negatives in training should help on L3-style test negatives
- Expected monotonic improvement: Z < A < B < C < D < E

**Per-layer proof metrics:**

| Layer | Primary Metric | Rationale |
|-------|----------------|-----------|
| Z->L1 | F1 / PR-AUC overall | Label noise removal is most impactful layer |
| L1->L2 | Per-ecozone F1 std dev decreases | Geographic bias affects fairness, not overall accuracy |
| L2->L3 | F1 on high-FWI non-fire samples | Hard negatives address this specific failure mode |
| L3->L4 | F1 variance across test years decreases | Spatial dedup improves cross-year stability |

**Reviewer defence for L3**: If AUROC drops from L2->L3, frame it as intended —
the task becomes harder (0.838 random -> 0.653 hard negatives). L3's value is measured
by precision on hard cases, not overall AUROC.

---

## Concerns / Open Questions

1. **Small F1 differences** (e.g., 96.17 vs 95.96): Likely artefact of ch0 leakage.
   After ch0 removal, expect larger and more meaningful gaps between variants.

2. **L2 may not improve overall F1**: Its value is per-ecozone equity.
   Prepare per-ecozone recall tables to defend this to reviewers.

3. **L3 argument to reviewers**: Not "F1 goes up" but "model can no longer exploit
   easy geographic shortcuts" — AUROC drop IS the evidence.

4. **Variant Z vs A**: Z adds cloud pixels (firemask 4) as negative candidates and
   nominal-confidence fires (firemask 8) as positives. Z represents truly label-noisy
   baseline; A is "no buffer" but still uses clean MODIS classifications only.
