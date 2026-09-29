# BODMAS / MalCSBSV Dataset Count Discrepancy Audit

**Project:** TRAMIF — Temporal-Reliability-Aware Multi-view Image Fusion  
**Date:** 29 September 2026  
**Status:** Measured dataset-audit result  
**Scope:** MalCSBSV baseline reproduction on BODMAS

## 1. Purpose

This audit compares the BODMAS sample counts reported by Zhang et al. for their MalCSBSV experiment with the samples currently present in the BODMAS metadata used by this project.

This work belongs to the baseline-reproduction stage described in Section 15.2 of the TRAMIF research framework. It does **not** modify the primary chronological TRAMIF protocol.

## 2. Source-paper cohort

Zhang et al. report using **18,498 BODMAS malware samples from 10 families** in the MalCSBSV experiment [1]. The per-family counts shown in their Figure 6(b) are:

| Family | Zhang et al. [1] |
|---|---:|
| ceeinject | 1,163 |
| drolnux | 920 |
| gandcrab | 939 |
| mira | 1,829 |
| musecador | 1,054 |
| sfone | 1,377 |
| sillyp2p | 1,526 |
| small | 3,217 |
| upatre | 2,988 |
| wabot | 3,485 |
| **Total** | **18,498** |

The paper states that stratified 10-fold cross-validation was used after dataset construction, but it does not provide a sufficiently explicit BODMAS sample-selection rule that reconstructs the 18,498-sample cohort from the current metadata.

## 3. Current BODMAS audit

The local metadata audit was executed on:

`E:/BODMAS_GW/bodmas_metadata.csv`

Measured properties of the current metadata:

- Total rows: **134,435**
- Rows with malware-family labels: **57,293**
- Rows without family labels: **77,142**
- Invalid SHA-256 identifiers: **7**
- Invalid timestamps: **0**
- Duplicate valid SHA-256 identifiers: **0**
- Labeled-malware time range: **2019-08-29 to 2020-09-30**

For the same 10 family names used by Zhang et al., the current metadata contains **23,318 unique valid SHA-256 samples**.

The 10-family candidate pool spans:

- **2019:** 4,768 samples
- **2020:** 18,550 samples
- **Total:** 23,318 samples

No sample from these 10 families occurs before 2019 in the current metadata. The earliest observed month is **August 2019** and the latest is **September 2020**.

## 4. Paper versus current BODMAS counts

| Family | Paper count | Current count | Difference | Difference (%) |
|---|---:|---:|---:|---:|
| ceeinject | 1,163 | 1,169 | +6 | +0.52% |
| drolnux | 920 | 920 | 0 | 0.00% |
| gandcrab | 939 | 957 | +18 | +1.92% |
| mira | 1,829 | 1,960 | +131 | +7.16% |
| musecador | 1,054 | 1,054 | 0 | 0.00% |
| sfone | 1,377 | 4,729 | **+3,352** | **+243.43%** |
| sillyp2p | 1,526 | 1,616 | +90 | +5.90% |
| small | 3,217 | 3,339 | +122 | +3.79% |
| upatre | 2,988 | 3,901 | **+913** | **+30.56%** |
| wabot | 3,485 | 3,673 | +188 | +5.39% |
| **Total** | **18,498** | **23,318** | **+4,820** | — |

The current metadata therefore contains **4,820 more samples** across these 10 families than the cohort reported in the paper.

The discrepancy is highly concentrated in two families:

- **sfone:** +3,352 samples
- **upatre:** +913 samples

Together, `sfone` and `upatre` account for **4,265 of the 4,820 extra samples, approximately 88.5% of the total discrepancy**.

## 5. Monthly distribution audit

A month-by-month audit was performed on the full, unfiltered 23,318-sample candidate pool.

### 5.1 Monthly totals

| Month | Total samples |
|---|---:|
| 2019-08 | 251 |
| 2019-09 | 1,151 |
| 2019-10 | 1,257 |
| 2019-11 | 689 |
| 2019-12 | 1,420 |
| 2020-01 | 1,567 |
| 2020-02 | 1,875 |
| 2020-03 | 2,203 |
| 2020-04 | 2,037 |
| 2020-05 | 1,743 |
| 2020-06 | 2,574 |
| 2020-07 | **2,576** |
| 2020-08 | 2,081 |
| 2020-09 | 1,894 |

The highest-volume month is **July 2020 with 2,576 samples**, followed very closely by **June 2020 with 2,574 samples**.

### 5.2 Peak month for each family

| Family | Peak month | Peak count |
|---|---|---:|
| ceeinject | 2020-08 | 334 |
| drolnux | 2020-07 | 457 |
| gandcrab | 2019-09 | 161 |
| mira | 2019-12 | 371 |
| musecador | 2020-01 | 357 |
| sfone | **2020-06** | **1,559** |
| sillyp2p | 2020-03 | 314 |
| small | 2020-03 | 499 |
| upatre | **2020-04** | **705** |
| wabot | 2020-08 | 723 |

The peaks occur in different months across families, rather than at one common dataset-wide cutoff.

### 5.3 Descriptive monthly spikes

A simple descriptive z-score diagnostic (`z >= 2`) was used to identify unusually high family-month counts. This is **not** treated as a formal statistical outlier test or as an exclusion rule.

The flagged family-month combinations were:

| Month | Family | Count | z-score |
|---|---|---:|---:|
| 2019-12 | mira | 371 | 2.18 |
| 2020-01 | musecador | 357 | 2.34 |
| 2020-03 | sillyp2p | 314 | 2.31 |
| 2020-04 | upatre | 705 | 2.18 |
| 2020-06 | sfone | **1,559** | **2.77** |
| 2020-07 | drolnux | 457 | 3.09 |
| 2020-08 | ceeinject | 334 | 2.35 |
| 2020-08 | wabot | 723 | 2.01 |

The most consequential spike for the reproduction discrepancy is `sfone`. Its current monthly distribution contains:

- 2020-05: **932**
- 2020-06: **1,559**
- 2020-07: **697**

Notably, **June 2020 alone contains 1,559 `sfone` samples, which is greater than the 1,377 `sfone` samples reported for the entire source-paper cohort**.

`upatre` also shows a substantial discrepancy. Its largest month is **April 2020 with 705 samples**, while the current total is 3,901 compared with 2,988 in the paper.

These measured distributions further indicate that the count discrepancy is family-specific rather than a simple uniform increase across BODMAS.

## 6. Timestamp-filter investigation

Several diagnostic month-removal experiments were performed before the full monthly distribution was inspected.

These filters changed the total sample count, but they did not reproduce the paper's per-family distribution. Removing months caused some families to fall below the paper counts while `sfone` and `upatre` remained substantially above them.

The full monthly audit strengthens that observation:

- family peaks occur in different months;
- `sfone` is concentrated particularly strongly in May–July 2020;
- `upatre` peaks in April 2020;
- other families peak at different points between September 2019 and August 2020.

Therefore, there is currently **no evidence that a single global month cutoff can reconstruct the source-paper cohort**.

Month-removal experiments are diagnostic only and are **not adopted as the official reproduction cohort**.

## 7. Interpretation

The measured result is that the current BODMAS metadata and the BODMAS cohort reported by Zhang et al. are not numerically identical, despite using the same 10 reported family names.

The discrepancy is also not evenly distributed. Approximately 88.5% of the total difference is explained by `sfone` and `upatre`, with `sfone` alone contributing 3,352 additional samples.

Possible explanations include:

1. the paper used a different BODMAS snapshot;
2. the authors applied an undocumented family-specific sampling procedure;
3. family labels or sample availability changed between dataset versions;
4. another preprocessing or inclusion rule was applied but not described in sufficient detail.

These are **hypotheses only**. The available source paper does not establish which explanation is correct.

The observed monthly spikes are descriptive properties of the current BODMAS metadata and should not be interpreted as evidence that those samples are erroneous or should be removed.

Importantly, the discrepancy should **not** be resolved by deleting months or samples until the total reaches 18,498. Doing so would fit the current dataset to the published number rather than reproduce a documented source procedure.

## 8. Decision for the project

For reproducibility, the project will retain the current **23,318-sample set as the documented candidate pool** for the 10 reported MalCSBSV families unless a source-supported selection rule is later identified.

The candidate pool is therefore defined as:

> All rows in the current BODMAS metadata with a valid SHA-256 and a normalized family label belonging to the 10 BODMAS families reported by Zhang et al.

No month-based exclusion is applied.

Any future attempt to construct an 18,498-sample approximation must be labeled explicitly as a **derived or diagnostic cohort**, not as an exact reproduction of the source-paper dataset.

This follows the reproducibility requirements in Sections 15.2–15.3 of the TRAMIF framework: preprocessing decisions, exclusions, manifests, and deviations from source procedures must be recorded rather than silently applied.

This baseline-reproduction cohort remains separate from the primary TRAMIF chronological experiment. The primary protocol continues to use the prespecified train/validation/future-test split defined in the research framework.

## 9. Conclusion

A reproducibility discrepancy exists between the BODMAS data currently available to this project and the BODMAS subset reported by Zhang et al.:

> **Zhang et al.: 18,498 samples**  
> **Current BODMAS metadata for the same 10 families: 23,318 samples**  
> **Difference: +4,820 samples**

The discrepancy is highly concentrated:

> **sfone: +3,352**  
> **upatre: +913**  
> **Combined: 4,265 / 4,820 ≈ 88.5% of the total difference**

The monthly audit also shows strong family-specific variation. The clearest case is `sfone`, which reaches **1,559 samples in June 2020 alone**, exceeding the paper's reported **1,377 `sfone` samples for its entire BODMAS cohort**.

The source paper does not provide enough information to uniquely reconstruct its 18,498-sample subset from the current metadata, and the measured monthly distributions do not support a simple global temporal cutoff as an explanation.

Accordingly, this difference is documented as a **baseline-reproduction limitation**. The current 23,318-sample candidate pool will be retained transparently rather than altered by arbitrary filtering.

## References

[1] J. Zhang, C. Guo, G. Shen, Y. Ping, Y. Cui, and Y. Chen, “A lightweight malware classification method based on short bit sequence visualization,” *Engineering Applications of Artificial Intelligence*, vol. 181, 115559, 2026.

[2] *Lightweight Malware Family Classification under Temporal Drift by Fusing Multiple Binary Image Views Based on Historical Reliability*, TRAMIF prospective research framework, Sections 15.2–15.3, 16 September 2026.
