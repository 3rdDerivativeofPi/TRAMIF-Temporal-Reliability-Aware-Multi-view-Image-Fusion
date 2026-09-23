# BODMAS Data Audit & Experiment Preparation — 23 Sep 2026

## Goal
Audit the downloaded BODMAS release before extraction/training, confirm that the malware archive, metadata, and disarm metadata are aligned, and decide which binary form should be used in the primary temporal-classification experiment.

## What was done

### 1. Inspected the BODMAS release
The local dataset currently contains:

- `BODMAS_disarmed_malware_binaries.zip`
- `bodmas_feature_vectors.npz.zip` → contains `bodmas.npz`
- `bodmas_metadata.csv`
- `meta_disarm.csv`
- `PE_modifier.py`
- `README.md`
- `example_arm.txt`

The malware ZIP contains files under `altered/<sha256>.exe`. These are the official **disarmed** binaries. `PE_modifier.py` shows that disarming changes two PE-header fields (`OPTIONAL_HEADER.Subsystem` and `FILE_HEADER.Machine`) to zero, while `meta_disarm.csv` stores the original values.

### 2. Ran the pre-freeze data audit
**Measured in this session:**

- Metadata rows: **134,435**
- Malware-labelled rows: **57,293**
- Blank-family / benign rows: **77,142**
- Malware archive members: **57,293**
- `meta_disarm.csv` rows: **57,293**
- Missing malware identifiers between metadata/archive/disarm metadata: **0**
- Duplicate archive paths: **0**
- Valid SHA-like archive filenames: **57,293 / 57,293**

The malware timestamps cover **Aug 2019–Sep 2020**, which matches the intended temporal experiment window.

### 3. Checked the training-only support rule
Using the framework's chronological protocol (Framework §11.2):

- **Training:** Aug 2019–Jan 2020
- **Validation:** Feb–Mar 2020
- **Future test:** Apr–Sep 2020

Training-only audit results:

- Training malware files: **20,333**
- Training family labels observed: **395**
- Families satisfying the provisional support rule (≥50 unique files across ≥3 training months): **51**

The **51 families are still provisional** until the family-label taxonomy is checked.

## Planned model architecture
The project uses three deterministic 64×64 binary-image views (Framework §4):

1. **Raw-byte image** — byte values arranged with width 256, then mask-aware resized to 64×64.
2. **Entropy image** — local Shannon entropy over 256-byte windows with stride 128, mapped back to byte offsets and resized to 64×64.
3. **SBSMI** — 6-bit short-state transition matrix, producing a fixed 64×64 Markov image.

Each view is processed by its own lightweight CNN branch. The SBSMI branch is intended to reproduce the MalSBSLCNet-style baseline first; raw-byte and entropy branches are added as complementary views (Framework §9). Final fusion operates on **calibrated class probabilities**, not on directly averaged embeddings (Framework §3.2, §7).

The intended implementation stack remains **PyTorch**, with PyTorch Lightning planned for the training workflow once the final MalSBSLCNet implementation details are confirmed.

## Important issues found

### 1. Disarmed vs. armed binaries
The downloaded binaries are not byte-identical to the original executable files because two PE-header fields were modified.

**Decision:** use the **official disarmed binaries for the primary experiment**.

Reason:
- all train/validation/future samples can use the same released byte representation;
- the classification pipeline does not require execution;
- re-arming is unnecessary for raw-byte, entropy, or SBSMI construction;
- changing to armed files would create a different corpus after the audit.

This limitation must be documented in preprocessing provenance. Results should be described as classification using the **BODMAS disarmed binary release**, not untouched original executables.

### 2. Possible family-label aliases / inconsistent spellings
The training-family audit showed suspiciously similar labels, e.g.:

- `gandcrab` / `grandcrab`
- `skeeyah` / `skeeeyah`
- `sytro` / `systro`
- `mira` / `miras`

These must **not** be merged just because they look similar. An authoritative BODMAS label/taxonomy check is required before freezing the eligible family set. This matters because alias resolution could change the current 51-family cohort.

### 3. Metadata anomalies outside the malware cohort
The full metadata file contains:

- **7 invalid SHA strings**
- **3,712 unparseable timestamps**

However, the malware identifier set is complete and aligned with all 57,293 binary files, so the observed anomalies appear to belong to the non-malware/benign records. They do not currently block the malware-family experiment, but they should remain documented.

### 4. Near-duplicate handling is not finished
Exact identifier alignment has been checked, but near-duplicate grouping still needs a conservative, training-specified policy before future testing (Framework §11.2). Later duplicate copies must not be moved backward into training.

## What has NOT been done yet

- No malware binaries have been extracted.
- No malware has been executed.
- `PE_modifier.py` has not been run.
- No files have been re-armed.
- No raw-byte, entropy, or SBSMI images have been generated from BODMAS yet.
- No BODMAS model training has been run.
- No validation/future performance has been measured.
- No future-test information has been used to tune the model.

Therefore, there are **no experimental performance results to report yet**.

## Next steps

1. Generate and preserve SHA-256 hashes for the downloaded source archives/files.
2. Audit the BODMAS family-label taxonomy and decide whether exact-string labels are the official policy.
3. Freeze the final eligible family set using **training data only**.
4. Create fixed `train`, `validation`, `future`, and `excluded` manifests.
5. Define the near-duplicate grouping policy before future testing.
6. Selectively extract only the required disarmed binaries.
7. Verify preprocessing on a small sample for raw-byte, entropy, and source-compatible SBSMI (Framework §4.4).
8. Reproduce the SBSMI + MalSBSLCNet baseline, then implement the three-view branches.
9. Use Feb–Mar only for calibration and reliability-rule selection, then freeze all components before Apr 2020 (Framework §7).
10. Run Apr–Sep as six untouched future test months.

## Current status
The **initial BODMAS access/timestamp/support audit is complete and passed basic integrity checks**. The main unresolved item before cohort freeze is the **family-label taxonomy**. The primary training corpus should remain the **official disarmed BODMAS binaries**.
