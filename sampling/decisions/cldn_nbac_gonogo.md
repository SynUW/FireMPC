# CLDN and NBAC Go/No-Go Decision

**Date:** 2026-04-04
**Status:** FINAL

---

## CLDN (Canadian Lightning Detection Network)

**Dataset:** Canadian Lightning Detection Network  
**Canada Open Gov Portal:** https://open.canada.ca/data/en/dataset/3e49c1dc-afe0-45ab-b16d-8da8d72e7b94

### Feasibility Assessment

HTTP HEAD request to Canada Open Government Portal on 2026-04-04:

```
curl -I --max-time 30 'https://open.canada.ca/data/en/dataset/3e49c1dc-afe0-45ab-b16d-8da8d72e7b94'
Result: HTTP/1.1 404 NOT FOUND
```

CKAN API call to retrieve resource URLs:

```
curl -s 'https://open.canada.ca/data/api/action/package_show?id=3e49c1dc-afe0-45ab-b16d-8da8d72e7b94'
Result: No output (dataset not found or not accessible)
```

The dataset page returns 404 and the CKAN API returns no data, indicating the CLDN dataset is not publicly available through this portal at this time.

**Decision: OUT OF SCOPE for v1**

**Reason:**
1. CLDN data accessibility check result: HTTP 404 NOT FOUND — dataset URL is not accessible; CKAN API returns no data for this dataset ID.
2. No programmatic bulk download available: The dataset is not reachable via the Canada Open Government Portal API at the tested endpoint.
3. Integration cost: Adding a new daily raster variable requires preprocessing, spatial alignment, temporal matching — estimated 2+ days of engineering work for a covariate that adds marginal value to the sampling strategy.
4. Sufficient without CLDN: The FWI components (ISI, BUI) already encode fire weather conditions relevant to ignition risk. CLDN would add ignition source information but is not required for the hard-negative scoring methodology.
5. Future work: CLDN data can be added as an enhancement (ENH-01 in v2 requirements) if the dataset becomes available through an alternate source.

**Impact on Phase 2:** None. Layer 3 hard-negative scoring uses only ERA5-derived FWI components.

---

## NBAC (National Burned Area Composite)

**Dataset:** National Burned Area Composite (Natural Resources Canada)

**Decision: OUT OF SCOPE for v1** (confirmed in STATE.md)

**Reason:**
1. Project decision recorded in STATE.md: "NBAC: Out of scope — removed from plan"
2. MOD14A1 is sufficient ground truth for sampling purposes: 2,176,363 high-confidence fire pixel-days across 2000–2024 provides robust positive labels.
3. NBAC would add characterization value (comparison figure) but not enough to justify timeline cost given the May 6 submission deadline.

**Impact on Phase 2:** None. MODIS MCD14A1 is the sole ground truth source.

---

## Summary

| Dataset | Decision | Reason |
|---------|----------|--------|
| CLDN    | Out of scope | HTTP 404 on portal; no bulk download available; engineering cost + marginal value; deferred to ENH-01 |
| NBAC    | Out of scope | Already decided; MODIS sufficient for v1 |
