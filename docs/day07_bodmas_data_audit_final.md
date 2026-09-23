# Day 07 — BODMAS Data Audit and Binary Access

**Date:** 23 September 2026

## Purpose

Finish the BODMAS pre-training data audit after obtaining the official disarmed
malware archive, and verify that the metadata, binary filenames, disarm metadata,
chronology, and provisional training-family cohort are internally consistent.

This document contains **dataset/cohort measurements only**. No classifier
performance has been measured yet.

---

## What was completed

### 1. Corrected metadata audit

The canonical audit script is now:

`audit_bodmas_pre_freeze_v2.py`

The earlier timestamp warning was traced to the parser rather than the dataset.
BODMAS contains mixed timestamp formats, so the correct parse is:

```python
pd.to_datetime(..., format="mixed", utc=True)
```

Measured by the v2 audit:

- total metadata rows: **134,435**
- unique sample identifiers: **134,435**
- duplicate identifier rows: **0**
- family-labelled malware rows: **57,293**
- blank-family / benign rows: **77,142**
- non-standard identifiers: **7**
  - malware: **0**
  - benign: **7**
- invalid timestamps: **0**
  - malware: **0**
  - benign: **0**

The seven non-standard IDs are benign records of the form
`VirusShare_<32-hex>` and are not part of the malware-family cohort.

Measured timestamp range:

- complete metadata: **2007-01-01 08:46:39 UTC** to
  **2020-09-30 03:05:27 UTC**
- malware only: **2019-08-29 09:59:25 UTC** to
  **2020-09-30 03:05:27 UTC**

Therefore, the previous result of `3712` unparseable timestamps is obsolete and
must not be reported as a BODMAS data problem.

### 2. Chronological protocol verified

The malware records divide exactly into the prespecified periods
(Framework §11.2):

| Period | Dates | Measured samples |
| --- | --- | ---: |
| Training | Aug 2019–Jan 2020 | **20,333** |
| Validation | Feb–Mar 2020 | **9,259** |
| Future test | Apr–Sep 2020 | **27,701** |
| **Total** | | **57,293** |

Malware samples outside these protocol periods: **0**.

The Apr–Sep count is recorded only as a cohort-audit quantity. Future labels or
future performance must not be used for training, calibration, reliability-rule
selection, or model selection.

### 3. Training-only family support verified

Using only Aug 2019–Jan 2020 and the provisional support rule from
Framework §11.2:

- at least **50 unique training files**
- observations in at least **3 training months**

the v2 run measured:

- training families: **395**
- provisionally eligible families: **51**
- eligible-family training files: **18,061 / 20,333**
- training coverage: **88.83%**
- excluded families: **344**
- excluded-family training files: **2,272**

Eligible-family month coverage:

- **45** families appear in all 6 training months
- **4** appear in 5 months
- **2** appear in 4 months

Support boundaries:

- smallest eligible family: `coinminer` — **50 files / 6 months**
- largest eligible family: `wacatac` — **3,311 files / 6 months**
- highest-support excluded family: `skeeyah` — **49 files / 6 months**

The generated `audit_eligible_families_v2.csv` contains **51 rows**, uses
contiguous class IDs **0–50**, and exactly matches the rows marked
`eligible_provisional=True` in `audit_training_family_support_v2.csv`.

The family set is still **provisional**, because similar label strings such as
`skeeyah` / `skeeeyah`, `gandcrab` / `grandcrab`, and `sytro` / `systro`
must be checked against an authoritative BODMAS taxonomy before any labels are
merged or the class set is frozen.

### 4. Disarmed binary archive verified

Measured archive properties:

- archive: `BODMAS_disarmed_malware_binaries.zip`
- archive file size: **72,932,177,367 bytes**
- binary members: **57,293**
- valid SHA-like binary filenames: **57,293 / 57,293**
- duplicate archive paths: **0**
- uncompressed binary bytes: **161,193,543,852**

`meta_disarm.csv` contains:

- rows: **57,293**
- unique SHA identifiers: **57,293**
- invalid SHA strings: **0**
- duplicate SHA rows: **0**

Cross-checking the three identifier sets produced:

- metadata malware IDs missing from archive: **0**
- archive IDs missing from malware metadata: **0**
- archive IDs missing from `meta_disarm.csv`: **0**
- `meta_disarm.csv` IDs missing from archive: **0**

This confirms that the malware metadata, disarmed binary archive, and disarm
records are internally aligned.

The audit only inspected the ZIP directory. The malware corpus was **not**
bulk-extracted or executed.

### 5. Feature-vector package verified

`bodmas_feature_vectors.npz.zip` contains:

- `bodmas.npz`
- uncompressed size: **724,345,841 bytes**
- compressed size: **596,121,598 bytes**

These feature vectors may later support feature-based baselines, but they cannot
replace the binary files required for the raw-byte, entropy, and SBSMI views.

---

## Binary policy

### Primary experiment: use the official disarmed binaries

The primary experiment will use **disarmed BODMAS bytes** consistently for:

- training
- validation
- future testing

`PE_modifier.py` will **not** be run for the primary experiment and samples will
not be re-armed.

The release script shows that disarming sets two PE-header fields to zero:

- `OPTIONAL_HEADER.Subsystem`
- `FILE_HEADER.Machine`

The paper must therefore describe the corpus as the **BODMAS disarmed binary
release**, not as untouched original executables.

Using the same released byte representation in every chronological period also
keeps preprocessing consistent with Framework §4.4.

---

## Representation and architecture status

The planned three views remain those in Framework §4:

1. **Raw byte**
   - byte layout width 256
   - deterministic mask-aware resize to 64×64

2. **Entropy**
   - 256-byte entropy windows
   - stride 128
   - overlap values mapped to byte offsets
   - same final 64×64 layout

3. **SBSMI**
   - non-overlapping 6-bit states
   - adjacent state-transition matrix
   - fixed 64×64 output
   - source-compatible bit order, tail handling, and quantization must remain
     deterministic

The project architecture is still a **planned multi-view model**, not a measured
result. Each view will have a lightweight classifier branch, and the final method
will fuse calibrated class probabilities using historical reliability
(Framework §3.2 and §7).

For the SBSMI source baseline, the current MalSBSLCNet specification includes:

- initial 3×3 convolution with 16 channels
- Batch Normalization
- six lightweight/inverted-residual-style modules
- adaptive average pooling
- classifier head

The exact internal block implementation must stay source-compatible rather than
being inferred from the phrase "inverted residual" alone.

The intended implementation stack remains **PyTorch / PyTorch Lightning**.

---

## Problems found

### Timestamp parsing

**Problem:** the first pre-freeze script incorrectly reported 3,712 unparseable
timestamps.

**Resolution:** v2 uses `format="mixed"` and measured **0 invalid timestamps**.
The v2 output is now the canonical timestamp audit.

### Possible family-label aliases

**Problem:** several family names look like spelling variants, and one candidate
(`skeeyah`) is immediately below the eligibility threshold.

**Resolution required:** verify the authoritative BODMAS label taxonomy. Do not
merge labels manually and do not freeze the 51-family class set until this is
resolved.

### Disarm transformation

**Problem:** the released binary bytes differ from the original armed PE files in
two header fields.

**Resolution:** use the official disarmed corpus consistently and document the
transformation. Any armed-vs-disarmed comparison would be a separate exploratory
sensitivity analysis rather than part of the primary protocol.

### Near-duplicate policy

**Problem:** exact ID alignment is clean, but the training-specified
near-duplicate grouping rule has not yet been frozen.

**Resolution required:** define the conservative grouping policy before the
future-test protocol is frozen, as required by Framework §11.2.

### Source provenance

**Problem:** cryptographic hashes for the downloaded top-level source packages
have not yet been recorded.

**Resolution required:** hash the downloaded archives/metadata and keep the hash
manifest with the experiment records.

---

## Generated audit artifacts

Canonical v2 outputs:

- `audit_summary_v2.json`
- `audit_training_family_support_v2.csv`
- `audit_eligible_families_v2.csv`

The v2 output supersedes the first pre-freeze audit where the two disagree,
especially for timestamp parsing.

---

## What has NOT been done

- no malware has been executed
- no sample has been re-armed
- `PE_modifier.py` has not been run
- the full 161 GB malware payload has not been bulk-extracted
- no real BODMAS raw-byte / entropy / SBSMI dataset has been generated yet
- no BODMAS model has been trained yet
- no validation metric has been reported
- no Apr–Sep future-test performance has been measured
- no future result has been used for tuning

Therefore there are **no classifier-performance findings yet**.

---

## Next steps

1. Generate and save SHA-256 hashes for the downloaded source packages.
2. Resolve the authoritative BODMAS family-label policy.
3. Freeze the final training-only eligible family set `K` and class-ID mapping.
4. Define and freeze the near-duplicate grouping policy.
5. Generate fixed train / validation / future / excluded manifests.
6. Selectively extract binaries required by the frozen manifests.
7. Verify raw-byte, entropy, and source-compatible SBSMI preprocessing on a
   small permitted sample.
8. Reproduce the SBSMI + MalSBSLCNet baseline.
9. Train the three view branches on Aug 2019–Jan 2020.
10. Use Feb–Mar only for calibration and historical reliability selection.
11. Freeze encoders, BatchNorm state, calibration, preprocessing, and fusion
    weights before April (Framework §7).
12. Evaluate Apr–Sep as six untouched future months.

---

## Status

**BODMAS metadata audit: complete.**

**BODMAS binary-release integrity audit: complete.**

The v2 audit is now the canonical audit record. The downloaded release is
internally aligned across metadata, binary filenames, and disarm metadata.

The main remaining blocker before cohort freeze is the **family-label taxonomy**.
The current **51-family set remains provisional** until that decision is made.
