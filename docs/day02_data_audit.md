# Day 02 BODMAS Data Audit

Date: 18 September 2026

## Purpose

Audit the publicly available BODMAS metadata before constructing the
chronological cohort for the temporal-reliability malware-family
classification experiment.

This audit concerns dataset structure and cohort construction only.
It contains no classifier-performance results.

## Source files

The public BODMAS release currently available locally contains:

- `bodmas.npz`
  - 2,381-dimensional feature vectors
  - binary labels: 0 = benign, 1 = malicious
- `bodmas_metadata.csv`
  - `sha`
  - `timestamp`
  - `family`
- `malware_catagory.csv`
  - `sha256`
  - `category`

According to the official dataset description, an empty `family` field
indicates a benign sample.

The NPZ binary target is therefore not the family-classification target used
by the primary experiment.

## Metadata audit

The audit script was:

`scripts/audit_bodmas_metadata.py`

Measured from the local public BODMAS metadata:

- Total records: 134,435
- Unique sample identifiers: 134,435
- Duplicate identifier rows: 0
- Family-labelled malware records: 57,293
- Records without a family label: 77,142

### Identifier validation

Seven identifiers are not 64-character SHA-256 strings.

All seven are benign records with identifiers in the form:

`VirusShare_<32-character hexadecimal value>`

No family-labelled malware sample has an invalid SHA-256 identifier.

These records are retained in the complete manifest through their original
`sample_id` rather than being silently discarded.

### Timestamp parsing

The metadata contains mixed timestamp formats.

Parsing with:

`pandas.to_datetime(..., format="mixed", utc=True)`

resulted in:

- Invalid timestamps: 0

The complete metadata spans:

- Earliest timestamp: 2007-01-01 08:46:39 UTC
- Latest timestamp: 2020-09-30 03:05:27 UTC

The older observations belong to the benign portion of the release and are
outside the primary malware-family cohort.

## Malware-only chronology

Measured for the 57,293 family-labelled malware samples:

- Earliest malware timestamp: 2019-08-29 09:59:25 UTC
- Latest malware timestamp: 2020-09-30 03:05:27 UTC
- Invalid malware timestamps: 0
- Malware samples without a valid SHA-256: 0

Using the prespecified chronological protocol:

| Period | Dates | Malware samples |
| --- | --- | ---: |
| Training | Aug 2019 - Jan 2020 | 20,333 |
| Validation | Feb 2020 - Mar 2020 | 9,259 |
| Future test | Apr 2020 - Sep 2020 | 27,701 |
| **Total** | | **57,293** |

All family-labelled malware records fall within one of the three protocol
periods.

The future-test sample count is recorded here only as a cohort-audit
quantity. Future-test labels and performance must not be used for training,
calibration, model selection, or reliability-rule selection.

## Training-only family support

Family eligibility was audited using only the training period.

Measured findings:

- Families represented in training: 395
- Families satisfying the provisional support rule: 51

The provisional rule requires:

- at least 50 unique training files, and
- samples in at least 3 distinct training months

This rule follows the prespecified cohort strategy and has not been adjusted
using validation or future-test performance.

## Generated artifacts

The audit generated:

- `data/manifests/bodmas_all_v0.csv`
- `data/manifests/bodmas_family_v0.csv`
- `data/manifests/bodmas_family_support_train_v0.csv`

The first two are reproducible sample-level manifests and are excluded from
Git.

The training-family support table is retained as a versioned derived research
artifact.

## Current limitation

Raw BODMAS malware binaries have not yet been obtained.

Therefore the primary raw-byte, entropy, and SBSMI image representations
cannot yet be constructed from real BODMAS malware samples.

Public metadata and feature vectors are sufficient for cohort auditing and
some later feature-based baselines, but not for the primary multi-view image
experiment.

## Status

Day-2 public-metadata audit: complete.

Next planned work while binary access is pending:

1. finalize chronology/support checks and tests;
2. begin deterministic preprocessing implementation using synthetic byte
   fixtures;
3. verify source-compatible SBSMI conventions before claiming reproduction.