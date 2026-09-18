# BODMAS Manifests

This directory contains derived metadata artifacts used to define the
chronological malware-family classification cohort.

The source dataset files are stored under:

data/raw/bodmas/

Raw dataset files are not tracked by Git.

## Generated manifests

### `bodmas_all_v0.csv`

Sample-level manifest derived from `bodmas_metadata.csv`.

Contains all public BODMAS metadata records, including benign and malware
samples.

Main fields include:

- `sample_id`: original identifier from the BODMAS `sha` column
- `sha256`: normalized SHA-256 where the identifier is a valid SHA-256
- `is_valid_sha256`
- `timestamp`
- `month`
- `family`
- `is_malware`
- `split`
- `binary_available`

This file is generated locally and is not tracked by Git.

### `bodmas_family_v0.csv`

Subset of `bodmas_all_v0.csv` containing samples with a non-empty malware
family label.

This is the candidate population for the primary malware-family
classification study.

Chronological split:

- Training: August 2019 - January 2020
- Validation: February 2020 - March 2020
- Future test: April 2020 - September 2020

This file is generated locally and is not tracked by Git.

### `bodmas_family_support_train_v0.csv`

Training-only family-support audit.

For each family appearing during the training period, the file records:

- number of unique training files
- number of distinct training months
- whether the family satisfies the provisional eligibility rule

The provisional eligibility rule is:

- at least 50 unique training files, and
- representation in at least 3 training months

Family eligibility is determined from the training period only.

This small derived artifact is tracked by Git.

## Regeneration

The manifests are produced by:

`scripts/audit_bodmas_metadata.py`

The script expects the public BODMAS metadata at:

`data/raw/bodmas/bodmas_metadata.csv`

Do not manually edit generated manifests. Changes to cohort construction
should be made in the generating script and documented separately.

## Raw binary availability

The public BODMAS release does not contain the raw malware binaries required
for the raw-byte, entropy, and SBSMI image representations.

Until restricted binary access is obtained, `binary_available` remains false.