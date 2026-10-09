# TRAMIF Engineering Log and New-Code Reference

**Reporting window:** approximately the 12 hours ending **10 October 2026, 02:47 (UTC+7)**, covering the group's image-generation refactor, distributed processing, optimization, and verification work.  
**Project:** *Lightweight Malware Family Classification under Temporal Drift by Fusing Multiple Binary Image Views Based on Historical Reliability*.  
**Research scope:** defensive binary-image representations and known-family malware classification.  
**Status:** **engineering implementation and limited verification**, **not** completed primary temporal-classification experiments.  
**Source basis:** the group GitHub `main` inspected during the conversation (commit `ecb0bef`), the new source patches supplied during the development discussion, existing framework document, a user-pasted single-binary comparison log, and the user-uploaded `comparison.csv` plus two 10-sample contact sheets.  
**GitHub:** https://github.com/3rdDerivativeofPi/TRAMIF-Temporal-Reliability-Aware-Multi-view-Image-Fusion

> **Result-language rule:** A reported measurement is identified below as either (A) an actual synthetic test run in this session or earlier discussion, or (B) a real-BODMAS run executed by a team member and furnished through terminal output or an uploaded CSV. The full private dataset has **not** been generated, merged, audited, trained, or evaluated in this report. Synthetic correctness tests and view-equivalence measurements are **not** malware-classification accuracy results.

## Executive summary

During this window we moved from a single-machine, weakly validated generation prototype toward a **distributed, versioned, chronological multi-view dataset pipeline**. Each laptop can execute the **same script** against the **same frozen manifest** while generating only its assigned modality (Raw-byte, Entropy, or SBSMI). Outputs use `split/month/modality/<original_sha>` and can be merged later only after sample IDs, source-content fingerprints, configurations, and output checksums pass verification. The production-ready status of this architecture remains **provisional until a complete, real-data generation and merge audit**.

Two **experimental faster implementations** were developed: vectorized 6-bit SBSMI and batched histogram-based local Entropy. They preserve the specified V2 representations on the synthetic regression suite. A user-run real-BODMAS comparison on **10 train-period binaries** independently found all images equal at the stored-data level, with substantial transformation-time reductions. This supports a carefully controlled speed-up, not an improvement in classification performance.

### Deliverable status at the end of the window

| Work item | Current evidence | Status |
|---|---|---|
| Inspect current group GitHub | Actual `main` branch inspected; legacy generator and dataset loader read | **Done** |
| Choose 3-laptop architecture | One manifest, one generator, modality-specific jobs, later merge | **Design agreed** |
| Build chronological split/month folders | V2 source and synthetic tests | **Implemented, synthetic-tested** |
| Build manifest + source fingerprinting + resumable generation | V2 source and tests | **Implemented, synthetic-tested** |
| Add dataset merge/audit/PyTorch loader | Source and tests | **Implemented, synthetic-tested** |
| Accelerate SBSMI/Entropy | Fast modules, regression tests | **Implemented, synthetic-tested** |
| Compare one real BODMAS binary | User-pasted terminal output | **PASS on one binary** |
| Compare 10 real BODMAS binaries | Uploaded `comparison.csv` and contact sheets | **10/10 PASS** |
| Complete train/validation images | No completed full-data generation log here | **Not established** |
| Merge complete outputs from all three actual laptops | No real full-dataset merge report here | **Not established** |
| Freeze and evaluate Apr–Sep 2020 | No frozen primary test report here | **Not run** |
| Merge these patches to group GitHub `main` | No commit/PR submitted by this assistant | **Not done** |

---

## 1. Why we changed the group generator

The inspected GitHub file `scripts/generate_multiview_dataset.py` already supported ZIP/folder sources, a progress bar, per-machine modality selection, trial runs, and basic CSV-based resumption. However, its main sample selection filtered **eligible family** without requiring the sample's **chronological split**. It wrote `raw_byte/<family>/<sha>.png`, `entropy/<family>/<sha>.png`, and `sbsmi/<family>/<sha>.png`, so training, validation, and future-test membership were not enforced by the output structure.

Other deficiencies: Raw-byte/Entropy loaded entire binaries and created file-sized intermediate arrays; the CSV resume path did not revalidate files or preprocessing versions; a failed row was added to the in-memory completed set; selection silently dropped missing/ambiguous source entries; every modality was written as uint8 PNG; and the generated files were not accompanied by reliable input/output content hashes.

### The critical preprocessing-version difference

The original group's `src/preprocessing/common.py` uses `torch.nn.functional.interpolate(..., mode="area")`. The V2 streaming Raw-byte/Entropy implementation uses **explicit fractional geometric-area overlap** between source and destination pixels, with padding masked out. These are **not numerically interchangeable in all cases**. Consequently:

- **V1 vs reference V2** is a *preprocessing-method comparison*; differences may be expected.
- **Reference V2 vs optimized V2** is an *equivalence comparison*; output values should agree under the documented tolerance.
- Preserve the repository's `configs/preprocessing_v1.yaml` and old outputs/checkpoints. Do **not** silently point V1 checkpoints to V2 inputs.
- Version all size, window, stride, bit-order, tail, quantization, masking, resize, storage, and code-hash settings (**framework §4.1–4.4**).

The group's streaming `src/preprocessing/implementation_sbsmi.py` already existed and was kept as the SBSMI **reference**. The high-performance SBSMI module is a separate experimental option, not a silent edit of that file.

---

## 2. Chronology and folder architecture

### 2.1 Immutable research split

| Split in code / folder | UTC observation timestamp | Research use |
|---|---|---|
| `train` | 2019-08-01 ≤ timestamp < 2020-02-01 | Train models; develop train-only historical episodes |
| `validation` | 2020-02-01 ≤ timestamp < 2020-04-01 | Temperature scaling, reliability estimation, rule/coefficient selection |
| `test` (`future_test` in older metadata) | 2020-04-01 ≤ timestamp < 2020-10-01 | Six **frozen**, prespecified monthly evaluations only |

**Source of chronology:** BODMAS observation timestamp, **not** the PE compilation timestamp. The 51-family label mapping is derived from training records only and read from `data/manifests/bodmas_eligible_families_v0.csv` (**framework §§5, 7, 11**).

### 2.2 Local and merged directory layout

```text
E:/TRAMIF_jobs/sbsmi/       # laptop A: same structure, only sbsmi outputs
E:/TRAMIF_jobs/raw/         # laptop B: only raw_byte outputs
E:/TRAMIF_jobs/entropy/     # laptop C: only entropy outputs

Each laptop root:
  config.json
  train/2019-08/sbsmi/<original_sha>.png    # example for SBSMI laptop
  train/2019-09/...
  validation/2020-02/...
  reports/<split>/<month>/<modality>/<original_sha>.json
  logs/<timestamp>_<modality>.csv
  logs/<timestamp>_<modality>.json

Coordinator's merged root:
  config.json
  merged_manifest.csv
  train/
    2019-08/{raw_byte,entropy,sbsmi}/<original_sha>.<ext>
    2019-09/ ... 2020-01/
  validation/
    2020-02/{raw_byte,entropy,sbsmi}/<original_sha>.<ext>
    2020-03/{raw_byte,entropy,sbsmi}/<original_sha>.<ext>
  test/                  # generated ONLY later under a confirmed frozen configuration
    2020-04/ ... 2020-09/
  reports/
```

The manifest is authoritative: a directory name alone never proves correct split membership. Outputs use `raw_byte` / `entropy` **64×64 float32 `.npy` in [0,1]** and **SBSMI 64×64 uint8 `.png`**. Optional Raw-byte/Entropy PNGs are **visual previews**, not the model's input data.

### 2.3 How the three laptops stay consistent

1. Coordinator builds **one canonical CSV manifest** and fixes the 51-family mapping. Send **identical files** to all authorized teammates.
2. Install the **same Git revision and V2 source files** on all laptops. Store a source-code/configuration fingerprint.
3. Member A generates `sbsmi`; member B generates `raw_byte`; member C generates `entropy`. Every modality runs from the same original-SHA-indexed binaries, independently.
4. Each output records `original_sha`, `split`, `month`, class ID, actual `disarmed_content_sha256`, CRC32, size, output SHA-256, and configuration ID.
5. Coordinator transfers **the whole job roots**, including reports/config, and merges by `(original_sha, split, modality)`—**never** by row order or image appearance.
6. Audit complete, correctly typed and matching views **before** training.

**Two distinct hashes:** original filename SHA is the **sample identifier** linking BODMAS labels/timestamps; SHA-256 computed from the **disarmed bytes** is the **input-integrity fingerprint**. They may legitimately differ because the disarming procedure modified the file content. The merger checks that every view for an original SHA has the **same disarmed-content SHA**.

---

## 3. Which code version should the team actually keep?

Several intermediate packages were produced during iteration. **Do not copy all packages blindly**; they contain duplicated or conflicting filenames.

| Iteration | What it added | Recommended treatment |
|---|---|---|
| Initial `TRAMIF_multiview_v2_patch` | First validated chronological V2 generator and audit, new fractional-area streamers | **Superseded** by the distributed generator; preserve only as an engineering history/reference |
| `TRAMIF_distributed_multiview_v2_patch` | Canonical manifest builder, per-laptop generator, merger, auditor, shared contract, temporal Dataset | **Base implementation to integrate** |
| `TRAMIF_sbsmi_speed_experiment` | Early isolated vectorized SBSMI plus tests | **Superseded** by fast-views bundle |
| `TRAMIF_fast_sbsmi_entropy_with_tests` | Fast SBSMI + Entropy modules, original-V2 reference modules, direct comparator, tests | **Optional optimized extension**, activate after equivalence verification |
| `TRAMIF_compare_10_old_fast_images` | 10-training-binary visual and numeric comparison script + tests | **Diagnostic utility**, not production dataset generation |

**Naming conflict:** The earlier single-machine V2 patch and the distributed patch both contain `scripts/audit_multiview_dataset.py` with different interfaces. Keep the **distributed** version, whose CLI uses `--root`, **not** the earlier one. The fast bundle duplicates `entropy_core.py`, `entropy_stream.py`, and `entropy_image_stream.py` as reference dependencies; keep just one consistent copy matching the tested distributed V2 files.

**No GitHub changes are implied:** these are proposal/implementation files delivered in chat, not evidence that a commit was pushed to the group repository.

---

## 4. New files and their functions: complete reference

This section explains the **current distributed V2 base**, **fast options**, and **comparison utilities**. Internal helper names begin with `_`, but they are still described because they affect correctness. Signatures are summarized; consult the `.py` source for detailed parameters/defaults.

### 4.1 `src/data/multiview_contract.py` — shared dataset rules

**Purpose:** the common contract imported by the generator, merger, auditor, and PyTorch Dataset. Ensures every program uses the same split dates, mapping, output paths, validation and reporting rules.

**Constants:**

- `MODALITIES`: `('raw_byte', 'entropy', 'sbsmi')`.
- `SPLITS`: `('train', 'validation', 'test')`.
- `SPLIT_RANGE`: immutable UTC time bounds shown in §2.1.
- `SHA_RE`: validates the 64-character lowercase hexadecimal original sample ID.

| Function | Behavior / reason |
|---|---|
| `file_sha256(path, chunk_size=...)` | Stream a file through SHA-256 to fingerprint manifests, mappings, outputs, etc.; doesn't load the file entirely. |
| `canonical_hash(value)` | Canonically JSON-serialize a configuration with stable key ordering, then hash it. A different algorithm/code/manifest config should produce a different ID. |
| `write_json_atomic(path, value)` | Write a temporary JSON file and replace the destination only after a complete write, reducing torn metadata. |
| `read_csv(path)` | Read CSV rows as dictionaries; reject an absent/empty header. |
| `write_csv(path, records, columns)` | Write a headered CSV to a temporary file and replace atomically. |
| `load_mapping(path)` | Validate the authoritative eligible-family CSV: **exactly 51 families, unique IDs 0–50**, no duplicate family names. |
| `normalize_split(value)` | Convert `future_test` alias to `test`; reject unknown split labels. |
| `parse_timestamp(text)` | Parse standardized ISO timestamp and derive UTC `YYYY-MM-DD` and `YYYY-MM`; downstream date/split validation uses this. |
| `load_manifest(path, mapping)` | Validate each sample's original SHA, unique ID, family/class mapping, timestamp, month, split, optional size/ZIP CRC; return deterministically sorted records. |
| `select_rows(rows, splits, months, limit, shard_id, num_shards)` | Select chronological and optional monthly subsets, apply **SHA-based deterministic sharding**, and cap the count **per selected split**. |
| `output_paths(root, row, modality)` | Construct image and JSON-report paths under `split/month/modality/<sha>`. SBSMI ends in `.png`; other views end in `.npy`. |
| `load_image(path, modality)` | Decode an SBSMI grayscale PNG or load a safe NPY array (`allow_pickle=False`), then validate it. |
| `validate_image(array, modality)` | Require 64×64; SBSMI `uint8`; Raw-byte/Entropy finite float32 within `[0,1]`. |
| `expected_report(row, modality, config_id)` | Build invariant fields that must match in each per-image report. |
| `valid_report(root, row, modality, config_id)` | Verify a cached output/report pair, required metadata, configuration ID, output checksum, dtype and shape. Used for resume, merge, audit. |

**Important:** integrity checks only work as well as the trusted input manifest and the original archive/index. An output hash confirms saved bytes; it does **not** by itself certify research-label truth or resolve near-duplicates.

### 4.2 `scripts/build_temporal_manifest.py` — canonical sample list

| Function | Behavior / reason |
|---|---|
| `main()` | Parse `--metadata`, `--mapping`, `--output` and optional `--index`. Read BODMAS metadata, normalize original SHA/family, keep 51 training-derived eligible families, parse mixed observation timestamps as UTC, assign chronological splits, attach optional size/CRC index fields, sort and save one manifest. It never reselects eligible families using validation/test accuracy. |

**Output columns:** `sha,family,class_id,split,month,timestamp,file_size,zip_crc32`.

**Call once** as the coordinator; all workers should use that exact file and its fingerprint. The fact that future rows are present in a manifest must **never** authorize tuning on future labels.

### 4.3 `scripts/generate_multiview_dataset_v2.py` — unified per-laptop generator

| Function / class | Behavior / reason |
|---|---|
| `normalized_source_digest(path)` | Hash source code after normalizing CRLF/LF line endings to keep equivalent Windows/Linux files comparable. |
| `build_config(manifest, mapping)` | Collect manifest/mapping hashes, source-module hashes, fixed V2 algorithms and data formats into one versioned configuration. |
| `ensure_output_config(out, config, splits, frozen_id, dry_run)` | Refuse a changed configuration in an existing job directory. Test generation requires a pre-existing historical config and exact `--frozen-config-id` (§7). |
| `build_source_index(path)` | Index archive members or extracted files by **original SHA in their filenames**; record byte sizes and ZIP CRC32 when available. |
| `select_sources(rows, index)` | Require exactly one nonempty matching source per selected manifest row; reject missing, ambiguous, or indexed-size/CRC conflicts before writing outputs. |
| `DigestReader.read(n)` | Wrap a binary stream; update SHA-256, CRC32 and byte count on the **bytes actually consumed** by each transform. |
| `DigestReader.fingerprint()` | Return recorded `disarmed_content_sha256`, `zip_crc32`, and `file_size`. |
| `open_source(source, mode, archive, member)` | Reopen the relevant ZIP entry or safely resolve an extracted file. Allows each modality to consume the same binary independently. |
| `verified_source(reader, row)` | Require consumed-byte size and available CRC32 to match the indexed source. |
| `make_image(reader, modality, size, chunk_size)` | Dispatch to Raw-byte, Entropy or SBSMI transformer; convert Raw/Entropy to float32 and validate output. |
| `save_image_atomic(path, array, modality)` | Write `.png` or `.npy` to a temporary path, then atomically replace the target. |
| `cached_or_none(out, row, modality, config_id)` | Revalidate an existing image/report on resume; return cache metadata if valid or request regeneration. |
| `parse_args(argv)` | Provide the CLI: source, manifest, mapping, output directory, modalities, splits, months, worker ID, sharding, limit, chunk size, overwrite, verification and test freeze options. |
| `run(args)` | Main orchestration: strict preflight → enforce configuration → log job metadata → process selected samples → verify saved images and report hashes → record `CREATED`, `VERIFIED`, or `ERROR` → fail if unresolved errors remain. |
| `main()` | CLI wrapper that prints failures and exits with a nonzero status on invalid inputs or generation errors. |

**Useful flags:** `--modalities`, `--splits`, `--months`, `--worker-id`, `--limit`, `--dry-run`, `--num-shards`, `--shard-id`, `--overwrite`, `--continue-on-error`, `--verify-source-on-skip`, `--save-previews`, `--frozen-config-id`.

**Semantics:** `--limit 5 --splits train validation` selects **5 samples from each split**, if available on that shard. `--num-shards 2 --shard-id 0/1` assigns disjoint SHA partitions. The completion CSV measures **per-file/modality generation duration**, which includes more than the isolated numerical transform time.

**Source-code fingerprint warning for the fast variant:** merely changing imports is **not enough**. Add the optimized modules to `SOURCE_MODULES` and document the selected algorithm/version. Then use a new output root/config ID. Every laptop must have the same updated source files, even if its assigned modality is Raw-byte.

### 4.4 `scripts/merge_multiview_dataset.py` — coordinator's strict merge

| Function | Behavior / reason |
|---|---|
| `build_parser()` | Define inputs, destination, manifest/mapping, selected splits/months/modalities, limits, overwrite, dry-run and frozen-test options. |
| `load_inputs_config(inputs, manifest, mapping)` | Require identical preprocessing configs from all job roots and matching authoritative manifest/mapping hashes. |
| `candidate_from_root(inputs, row, modality, config_id)` | Find **exactly one** valid image/report candidate for a SHA/modality across the inputs; reject missing or duplicate candidates. |
| `atomic_copy(source, destination, overwrite)` | Copy to a temporary file, verify its SHA-256, then move into place; refuse unexpected conflicts unless overwrite is explicitly enabled. |
| `merge(args)` | Full preflight every requested `(sample, view)`; require matching disarmed-content SHA/size/CRC across all modalities, then copy images/reports and write `merged_manifest.csv`. Future-test merging must be separate and frozen. |
| `main()` | Parse arguments and exit nonzero if merge validation fails. |

**Why separate merge?** A high image count can hide missing views. This tool refuses to turn an incomplete three-laptop job into an apparently complete training dataset.

### 4.5 `scripts/audit_multiview_dataset.py` — coverage/integrity audit

| Function | Behavior / reason |
|---|---|
| `build_parser()` | Define root, manifest, mapping, split/month/modality filters, trial limit and `--check-extras`. |
| `audit(args)` | Revalidate image/report pairs; count verified images per split/month/modality; check matching content SHA across views; report missing/corrupt outputs and suspicious exact cross-split duplicates. |
| `main()` | Print PASS/FAIL and exit nonzero for missing/invalid requested outputs. |

**Caveat:** exact duplicate content across train/test is a diagnostic warning requiring review, **not** authorization to tune a duplicate-removal rule after inspecting future outcomes. Near-duplicate detection (e.g., TLSH) remains a separately prespecified policy (**framework §11.2**).

### 4.6 `src/data/temporal_multiview_dataset.py` — PyTorch input

| Method | Behavior / reason |
|---|---|
| `TemporalMultiViewDataset.__init__(root, manifest, mapping, split, month, modalities, allow_test, return_sha)` | Verify manifest/mapping fingerprints and the split's file existence. Explicitly block test loading unless `allow_test=True`. Keep chronology independent of 10-fold reproduction code. |
| `__len__()` | Return selected sample count. |
| `__getitem__(index)` | Load requested modalities for one original SHA, convert SBSMI to float32 `/255`, keep Raw/Entropy already-normalized float32, add channel dimension, return `(views_dict, class_id)` or `(views_dict, class_id, sha)`. |

A batch of 32 would contain `views['raw_byte']`, `views['entropy']`, and `views['sbsmi']`, each `[32,1,64,64]`; labels are integers in `0..50`. **Run the complete dataset audit first:** this loader checks existence at initialization, but does not recalculate output SHA-256 on every training batch.

### 4.7 `src/preprocessing/raw_byte_stream.py` — Raw-byte V2, bounded memory

| Function | Behavior / reason |
|---|---|
| `_accumulate_block(...)` | Convert complete 256-wide layout rows to partial sums/counts for 64×64 geometric-area resizing. Correctly handle a partially valid final row; padding never contributes. |
| `stream_to_raw_byte(stream, file_size, chunk_size=65536)` | Read binary bytes incrementally, map byte values to width-256 rows, accumulate area weights, divide by 255, and return a normalized 64×64 float64 array; caller casts to float32 for storage. |
| `file_to_raw_byte(path, chunk_size=65536)` | Convenience file wrapper; check file size/mtime before and after processing to detect concurrent input modification. |

**Status:** streaming V2 Raw-byte exists and is covered by synthetic tests, but the team's new **10-sample reference-vs-fast trial did not compare a separate optimized Raw-byte implementation**. No Raw-byte speedup may be inferred from those SBSMI/Entropy timing columns.

### 4.8 `src/preprocessing/entropy_core.py` — Entropy's definition

| Function | Behavior / reason |
|---|---|
| `shannon_entropy(data)` | Compute byte-distribution Shannon entropy in bits (`0..8`) for one nonempty window using byte frequencies. |
| `local_entropy_values(data, window_size=256, stride=128)` | Non-streaming reference/helper: compute per-byte normalized `H/8` and mean over overlapping windows; useful to validate streaming edge cases. |

The exact entropy-window/tail convention is part of the experiment specification (**framework §4.2, §4.4**). The function's numbers describe byte diversity, not detection confidence or malware-class probabilities.

### 4.9 `src/preprocessing/entropy_stream.py` — reference V2 entropy per byte

| Function / method | Behavior / reason |
|---|---|
| `_BufferedInput.__init__(stream, file_size, chunk_size)` | Set up bounded input buffering and a declared file length. |
| `_BufferedInput.read_exact(count)` | Return the exact requested count from buffered reads, failing on early EOF. |
| `_BufferedInput.finish()` | Verify the declared bytes were fully consumed and that no unexpected extra byte remains. |
| `iter_local_entropy(stream, file_size, window_size=256, stride=128, chunk_size=65536)` | Yield ordered float64 blocks of finalized normalized entropy estimates, one value per real input byte; average entropy over every overlapping window and include the defined final partial window. |

This is the **slow/reference V2** against which the optimized entropy iterator is compared; do not delete it.

### 4.10 `src/preprocessing/entropy_image_stream.py` — reference V2 Entropy image

| Function | Behavior / reason |
|---|---|
| `_add_rows(...)` | Accumulate per-byte entropy values into 64×64 fractional-area sums/counts, excluding padded offsets. |
| `stream_to_entropy_image(stream, file_size, chunk_size=65536)` | Drive the reference per-byte iterator, reshape via 256-wide layout, normalize masked geometric-area averages and return a 64×64 float64 array. |
| `file_to_entropy_image(path, chunk_size=65536)` | Convenience file wrapper with before/after stat checks. |

Output is finally cast to float32 by the distributed generator. This module is the **reference V2 resize**, not the legacy PyTorch-area V1 resize.

### 4.11 `src/preprocessing/implementation_sbsmi.py` — existing group's reference, NOT a new replacement

The group's existing production reference is kept for regression testing. Key functions include `iterate_lbit_states`, `stream_to_sbsmi`, `file_to_sbsmi`, `zip_entry_to_sbsmi`, `validate_sbsmi`, and `save_sbsmi`.

`stream_to_sbsmi` reads MSB-first **nonoverlapping 6-bit** states across chunk boundaries, retains the final partial state using the locked high-order-zero convention, counts transitions from previous state to current state, row-normalizes a 64×64 matrix, and quantizes as `floor(255 × probability)` to `uint8`. The new vectorized version must match this reference's **4,096 pixels exactly**.

### 4.12 `src/preprocessing/sbsmi_vectorized.py` — optimized SBSMI

| Function | Behavior / reason |
|---|---|
| `stream_to_sbsmi_fast(stream, bit_num=6, chunk_size=65536)` | For 6-bit states, vectorize **three input bytes → four consecutive 6-bit states**, use `np.bincount` to accumulate the 64×64 transition matrix, carry incomplete triples/chunk transitions correctly, and preserve original floor quantization. For other bit widths, fall back to existing reference implementation. |

Its speed is due to replacing a Python per-byte/per-transition loop with NumPy array operations; it is **not** a new malware-representation definition. Use the reference's exact pixels as the acceptance criterion.

### 4.13 `src/preprocessing/entropy_stream_fast.py` — optimized entropy per byte

| Function | Behavior / reason |
|---|---|
| `_histograms_for_blocks(data)` | Compute byte-frequency histograms for a batch of **128-byte input blocks** using NumPy instead of rebuilding each overlapping 256-byte window from scratch. |
| `_normalized_entropy(counts)` | Compute `H/8` from frequency histograms using vectorized `count*log2(count)` arithmetic. |
| `_process_histograms(histograms, previous_histogram, previous_entropy)` | Combine adjacent block histograms into 256-byte windows, handle cross-chunk state and finalize the per-byte average of overlapping window entropies. |
| `iter_local_entropy_fast(stream, file_size, window_size=256, stride=128, chunk_size=65536)` | Emit float64 entropy values in original byte order using bounded input buffers; preserve one trailing partial window and strict file-size checks. For nonstandard window/stride parameters, defer to the reference iterator. |

**Key optimization:** preserve identical definitions while reducing repeated per-window Python overhead. The output may differ by approximately machine epsilon before float32 conversion because floating-point summation order changes.

### 4.14 `src/preprocessing/entropy_image_stream_fast.py` — optimized Entropy image

| Function | Behavior / reason |
|---|---|
| `_add_rows(...)` | Same masked fractional-area accumulation as reference V2, used to form the 64×64 output. |
| `stream_to_entropy_image_fast(stream, file_size, chunk_size=65536)` | Connect `iter_local_entropy_fast` to the unchanged 256-wide layout and geometric-area resize, producing a float64 image in `[0,1]`. |
| `file_to_entropy_image_fast(path, chunk_size=65536)` | File wrapper with source mutation checks. |

This module is the one the optimized Entropy worker should import **after** the numerical equivalence gate, with its source hash recorded.

### 4.15 `scripts/compare_fast_views.py` — direct, one-binary comparator

| Function | Behavior / reason |
|---|---|
| `open_binary(file_path, zip_path, member)` | Open one extracted binary or one exact ZIP member without extraction; expose its byte size. |
| `compare(file_path, zip_path, member, chunk_size, entropy_atol)` | Hash the original binary bytes, reopen the same source for reference and fast transforms, ensure each consumes all identical bytes, compare SBSMI for exact equality and Entropy for a numeric tolerance, and return a detailed dictionary with elapsed transform times. |
| `main(argv)` | CLI; print PASS/FAIL and optionally write a JSON report with `--report`. |

The comparator has a small internal `FingerprintReader`/`run` helper to guard against comparing the outputs of different source bytes. It does **not** create comparison PNG sheets; use `generate_compare_10.py` for that.

### 4.16 `scripts/generate_compare_10.py` — ten-sample diagnostic and contact sheets

| Function / class | Behavior / reason |
|---|---|
| `select_training_samples(csv_path, mapping_path, is_manifest)` | Validate the 51-family mapping, use only Aug 2019–Jan 2020 records, reject invalid/duplicate IDs or a manifest contradicting timestamps, sort candidates by original SHA. |
| `source_index(source_path)` | Discover original-SHA-named ZIP members or extracted files and their lengths. |
| `open_source(source_path, mode, member)` | Open the selected ZIP entry or safe relative disk file. |
| `HashingReader.read(size)` | Hash and count bytes consumed by a transform; used to prove the four transforms on a sample all used the same binary. |
| `run_transform(source_path, mode, member, size, implementation, modality, chunk_size)` | Run a chosen reference/optimized SBSMI or Entropy transform, time **that transform**, verify byte count and return array + actual content digest + time. |
| `write_view(output_root, sha, implementation, modality, array)` | Save SBSMI uint8 PNGs and Entropy float32 NPYs, verify read-back, and create an 8-bit Entropy preview PNG. |
| `make_contact_sheet(output_root, samples, modality)` | Produce a sheet of ten horizontally paired images: **Reference V2 left, Optimized V2 right**; include original SHA prefix and sample month. |
| `compare_10(source, csv_file, mapping, is_manifest, output, limit, max_size_mib, chunk_size, atol)` | Select the first `limit` eligible, available training binaries (optionally ≤ size cap), generate four outputs per binary (2 modalities × 2 implementations), compare arrays and produce `comparison.csv` plus both contact sheets. |
| `main(argv)` | Parse PowerShell CLI options for ZIP/folder, metadata/manifest, mapping, output, sample limit, size cap, chunk size, tolerance. |

**Sampling caveat:** it selects the **first SHAs in sorted order**, not a statistically representative or stratified random sample. It skips unavailable files for this **smoke test only**; the strict full generator/auditor must not silently exclude such samples. `--limit 10` means **10 binaries**, producing **20 SBSMI PNGs and 20 Entropy NPY arrays**, plus **20 Entropy preview PNGs** and **2 sheets**.

### 4.17 Added test files — what each checks

**`tests/test_distributed_multiview_v2.py`** (12 test functions, synthetic fixtures):

| Test | Contract exercised |
|---|---|
| `test_three_laptops_merge_and_dataloader` | Simulated three independent modality jobs, merge correctness, PyTorch Dataset loading. |
| `test_zip_folder_identical_and_cache_regenerates` | ZIP/folder byte equivalence and bad-cache replacement. |
| `test_merge_rejects_missing_modality` | Missing view cannot be silently merged. |
| `test_merge_rejects_different_source_bytes` | Same original ID with different disarmed bytes is rejected. |
| `test_future_test_guard_and_merge` | Enforces frozen-ID test-only generation/merge. |
| `test_sharding_assigns_disjoint_samples` | SHA shards do not overlap. |
| `test_invalid_split_date_is_rejected` | Split/date mismatch is fatal. |
| `test_manifest_builder_from_raw_metadata` | Canonical manifest creation from metadata. |
| `test_stream_algorithms_chunk_boundary_independence` | Chunk size does not change representations. |
| `test_missing_binary_preflight_no_output` | Missing input fails before partial dataset creation. |
| `test_conflicting_laptop_code_configs_rejected` | Different algorithm/code fingerprints cannot be combined. |
| `test_duplicate_modality_sources_rejected` | Duplicate views from multiple roots are rejected. |

**`tests/test_fast_views.py`** (parameterized suite generating 224 test cases in the development run): checks SBSMI exact pixels; entropy float64 tolerance and float32 identity; many short/long/tail/constant/random patterns; chunk-size independence; nonstandard fallback behavior; erroneous input lengths and invalid arguments; PNG/NPY round trip; one-binary comparison via ZIP and folder. Test function names include `test_entropy_image_matches_v2_reference`, `test_sbsmi_uint8_pixels_identical`, `test_entropy_chunk_independence`, `test_sbsmi_chunk_independence`, `test_entropy_patterns_and_per_offset_values`, `test_sbsmi_patterns`, `test_nondefault_entropy_window_delegates_to_reference`, `test_other_sbsmi_widths_delegate_to_reference`, `test_empty_input_fails`, `test_entropy_rejects_short_and_long_streams`, `test_entropy_invalid_args`, `test_png_npy_roundtrip_does_not_change_pixels`, and `test_compare_command_on_file_and_zip`.

**`tests/test_generate_compare_10.py`** (four synthetic tests): `test_compare_zip_images_and_contact_sheets` verifies four per-sample transforms and saved sheets, `test_compare_extracted_folder` checks folder mode, `test_manifest_timestamp_guard` rejects incorrect chronology, and `test_requires_enough_train_samples` rejects insufficient candidates.

**Session re-test:** On an assembled local overlay containing the distributed patch, fast patch, comparison patch, and the group's uploaded `implementation_sbsmi.py`, this session executed:

```powershell
python -m pytest -q tests/test_distributed_multiview_v2.py tests/test_fast_views.py tests/test_generate_compare_10.py
```

**Actual result: `240 passed in 3.51s`**, on deterministic synthetic fixtures. This does not include full-BODMAS, GPU, full-dataset merge, or model-performance testing.

### 4.18 Historical/superseded V2 prototypes and documentation

Earlier in the same engineering window, `TRAMIF_multiview_v2_patch` created a different `scripts/generate_multiview_dataset.py` (split-aware prototype), a different `scripts/audit_multiview_dataset.py`, and `src/preprocessing/sbsmi_stream_v2.py` (`stream_to_sbsmi_exact`). These established manifest checks, fractionally weighted streaming, and initial regression tests. **They were superseded by the distributed script and the existing group SBSMI reference**; do not overlay that old generator/auditor on the distributed patch. Its `tests/test_generate_multiview_dataset_v2.py` covers 12 foundational checks and `docs/multiview_v2_README.md` documents that prototype.

The standalone `TRAMIF_sbsmi_speed_experiment` introduced `src/preprocessing/sbsmi_vectorized.py` and `tests/test_sbsmi_vectorized.py` as a preliminary vectorization demonstration; the later fast-views package supplies the integrated version/tests. The `TRAMIF_DISTRIBUTED_MULTIVIEW_V2.md` guide documents install/generate/merge/audit commands. The fast-views and compare-10 `README.md` files document their respective optional tooling. All of these are **documentation/source artifacts, not measurement outputs**.

---

## 5. Actual real-BODMAS verification during the window

### 5.1 One-binary direct test (team member's pasted console log)

- ZIP member: `altered/ffffda51...e611.exe`.
- Binary size: **13,226,174 bytes**.
- Original filename SHA and disarmed-content SHA differ **as expected**.
- SBSMI: **PASS**, zero differing pixels; reference **11.5050 s**, optimized **0.3502 s** (about **32.9×**).
- Entropy: **PASS**, maximum array difference **1.39×10^-17**, zero differing float32/preview values; reference **3.0918 s**, optimized **0.5286 s** (about **5.85×**).

These are elapsed transformation timings from the team member's command-line run, **not** an independent run on this assistant's computer.

### 5.2 Ten-binary real BODMAS comparison (uploaded `comparison.csv`)

**Origin:** user-uploaded `comparison.csv` from `generate_compare_10.py`; both contact sheets were supplied by the user. The CSV has **10 rows and 14 columns**. All entries have `split=train` and all 10 have `status=PASS`. The measured files span **33,132 to 5,271,488 bytes**, totaling **15,110,552 bytes** (~14.41 MiB). Months represented: 2019-09, 2019-10, 2019-11, 2019-12, and 2020-01; the sample does **not** include 2019-08.

| Result | Recorded value |
|---|---:|
| Verified train samples | **10 / 10 PASS** |
| SBSMI differing pixels per sample | **0 for all ten** |
| Entropy float32 differing values per sample | **0 for all ten** |
| Entropy 8-bit-preview differing pixels per sample | **0 for all ten** |
| Maximum uncast Entropy absolute difference | **5.5511151231258×10^-16** |
| Sum of SBSMI reference transformation times | **13.132142 s** |
| Sum of SBSMI optimized transformation times | **0.226895 s** |
| **SBSMI aggregate time ratio** | **57.88× faster** on these samples |
| Sum of Entropy reference transformation times | **4.445937 s** |
| Sum of Entropy optimized transformation times | **0.750128 s** |
| **Entropy aggregate time ratio** | **5.93× faster** on these samples |

**Definition of aggregate speedup:** sum of original transform durations divided by sum of optimized transform durations for the same 10 binaries. This is **not** the average of individual speedups, not a latency guarantee, and not a full image-generation pipeline measurement. ZIP opening, repeated source reads, output I/O, report hashing and cross-machine transfer can reduce end-to-end gains.

#### Per-sample audit (derived from the uploaded CSV)

| Original SHA prefix | Month | File MiB | SBSMI ratio | Entropy ratio | SBSMI differing pixels | Entropy max abs difference |
|---|---|---:|---:|---:|---:|---:|
| `000093dfd22a` | 2019-10 | 0.983 | 53.0× | 6.7× | 0 | 4.4e-16 |
| `000129f62b2f` | 2020-01 | 0.096 | 38.7× | 6.6× | 0 | 3.3e-16 |
| `0002e5e1b878` | 2019-10 | 0.037 | 24.8× | 4.4× | 0 | 2.2e-16 |
| `0003fd12e07c` | 2019-12 | 0.233 | 36.6× | 5.2× | 0 | 4.4e-16 |
| `0004b5b986eb` | 2019-12 | 0.079 | 28.2× | 14.2× | 0 | 5.6e-16 |
| `0007bac2c646` | 2019-10 | 0.881 | 37.0× | 10.4× | 0 | 5.6e-16 |
| `000a0dcbb0e1` | 2020-01 | 3.947 | 75.6× | 5.4× | 0 | 5.6e-17 |
| `000a6826b216` | 2019-11 | 5.027 | 59.3× | 4.7× | 0 | 0 |
| `000a82358156` | 2019-12 | 3.095 | 52.7× | 6.6× | 0 | 3.3e-16 |
| `0017f1c71775` | 2019-09 | 0.032 | 28.9× | 2.9× | 0 | 2.2e-16 |

The test purpose was **representation identity on selected existing train binaries**. Selecting the first 10 sorted SHA IDs and capping their size is convenient for troubleshooting but **not representative** of malware families, months, full file-size distribution or future drift. The numbers must not be described as model F1, detection accuracy, family recall, or a population-wide speed guarantee.

### 5.3 What the contact sheets mean

- **Entropy sheet:** broad, horizontal gray/light/dark bands reflect local byte entropy after offset mapping and vertical downsampling. Right and left columns appear the same; the array-level checks are the decisive evidence.
- **SBSMI sheet:** most pixels are near black, with scattered brighter transitions. This can be expected because the 64×64 transition probabilities are often small and their floor-quantized 0–255 intensities have a low dynamic range. A dark image is **not** proof of incorrect generation or automatically evidence of good classification.
- The reported exact saved-data agreement is stronger than visual similarity. **Neither sheet proves that the files are correctly labeled malware families**; classification validity requires independent labels, proper temporal evaluation and model experiments.
- If worried about an **entirely zero** SBSMI matrix, inspect each actual 64×64 PNG for `min`, `max`, `np.count_nonzero` and `np.unique`; do not rely on the enlarged contact-sheet rendering alone.

### 5.4 Why entropy's tiny float64 differences do not change stored data here

Reference and batched entropy compute the same mathematical formula with potentially different floating-point addition order. In these ten runs the largest pre-cast absolute difference was ~`5.55e-16`. After converting to **float32 NPY**, all 4,096 values per sample were identical for the compared implementations, and so were the 8-bit preview pixels. The output guarantee applies **only to the tested samples and configuration**; keep the real-file comparison and numeric tolerance checks as regression tests for future optimizations.

---

## 6. How to run the current workflow (Windows PowerShell)

Commands assume the **distributed V2** source and optional **fast** modules have been copied into the group repository and their imports/fingerprints reconciled. They are **not** guaranteed to work against the untouched legacy `main` branch. Use a feature branch and inspect `git diff` before committing.

### A. Tests first

```powershell
python -m pytest -q tests/test_distributed_multiview_v2.py tests/test_fast_views.py tests/test_generate_compare_10.py
```

### B. Coordinator builds and distributes one canonical manifest

```powershell
python -m scripts.build_temporal_manifest `
  --metadata "E:\BODMAS_GW\bodmas_metadata.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output "data\manifests\multiview_v2_manifest.csv"
```

Use the **same exact file and mapping** on every laptop; never regenerate a different list per member.

### C. Each member runs an initial split-aware smoke test

```powershell
# Laptop A; use sbsmi
python -m scripts.generate_multiview_dataset_v2 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "E:\TRAMIF_jobs\sbsmi" `
  --modalities sbsmi --splits train validation `
  --worker-id laptop-A --limit 5 --dry-run
```

Review the selection, then remove `--dry-run`. On other laptops, switch respectively to `--modalities raw_byte --output-dir "E:\TRAMIF_jobs\raw"` or `--modalities entropy --output-dir "E:\TRAMIF_jobs\entropy"` and choose distinct `--worker-id`. For full generation, remove `--limit 5`. `--limit` applies per selected split.

### D. Dry-run and execute the merge

```powershell
python -m scripts.merge_multiview_dataset `
  --inputs "E:\TRAMIF_collected\sbsmi" "E:\TRAMIF_collected\raw" "E:\TRAMIF_collected\entropy" `
  --output-dir "E:\TRAMIF_collected\merged" `
  --splits train validation --limit 5 --dry-run
```

Once complete, remove `--dry-run`. After a full run remove `--limit 5`. The merge should fail loudly if anything is missing or inconsistent.

### E. Audit merged data

```powershell
python -m scripts.audit_multiview_dataset `
  --root "E:\TRAMIF_collected\merged" `
  --splits train validation --check-extras
```

### F. Compare original V2 and optimized V2 on one real training binary

```powershell
python -m scripts.compare_fast_views `
  --zip "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --member "altered/YOUR_TRAIN_SHA.exe" `
  --report "reports\fast_comparison.json"
```

Use the actual ZIP member's internal path (typically forward slashes: `altered/YOUR_TRAIN_SHA.exe`). This command compares the mathematical outputs but **does not save** two contact sheets.

### G. Generate the 10-sample side-by-side images

```powershell
python -m scripts.generate_compare_10 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --metadata "E:\BODMAS_GW\bodmas_metadata.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "data\derived\compare_old_fast_10" `
  --limit 10 --max-size-mib 20

# View in Windows
 ii "data\derived\compare_old_fast_10\entropy_comparison_sheet.png"
 ii "data\derived\compare_old_fast_10\sbsmi_comparison_sheet.png"
```

**Comparator output:** `comparison.csv`, `entropy_comparison_sheet.png`, `sbsmi_comparison_sheet.png`, individual `reference_v2/` and `fast_v2/` files. This is **SBSMI and Entropy only**: a 10-binary trial does **not** generate Raw-byte pairs.

### H. Minimal PyTorch test

```python
from torch.utils.data import DataLoader
from src.data.temporal_multiview_dataset import TemporalMultiViewDataset

dataset = TemporalMultiViewDataset(
    root=r"E:\TRAMIF_collected\merged",
    manifest="data/manifests/multiview_v2_manifest.csv",
    mapping="data/manifests/bodmas_eligible_families_v0.csv",
    split="train",
)
views, labels = next(iter(DataLoader(dataset, batch_size=32)))
assert views["raw_byte"].shape[1:] == (1, 64, 64)
assert views["entropy"].shape[1:] == (1, 64, 64)
assert views["sbsmi"].shape[1:] == (1, 64, 64)
```

### I. Quick darkness/zero-value diagnostic for SBSMI (optional)

```python
from pathlib import Path
from PIL import Image
import numpy as np

png = Path(r"data\derived\compare_old_fast_10\fast_v2\sbsmi\REPLACE_WITH_SHA.png")
image = np.asarray(Image.open(png).convert("L"))
print("shape:", image.shape, "min:", image.min(), "max:", image.max())
print("nonzero:", np.count_nonzero(image), "out of", image.size)
print("unique values:", np.unique(image)[:25])
```

---

## 7. Remaining tasks and release gate

| Priority | Task | Acceptance condition |
|---|---|---|
| P0 | Reconcile source files with current group GitHub | Controlled feature branch; no accidental overwrite of V1 or conflicting auditor files |
| P0 | Decide/document exact V2 resize and storage | Locked config, version and explicit rationale (§4.4) |
| P0 | Update fast imports and `SOURCE_MODULES` consistently on **all laptops** | Identical config IDs and file hashes before distributed generation |
| P0 | Run small real-data production smoke tests on **all three modalities** | Correct split/month paths, original SHA, content fingerprints, types and saved-file checks |
| P0 | Test actual three-machine transfer and merge | One candidate per `(SHA, modality)`; hashes align; audit PASS |
| P0 | Complete chronological train/validation build and full audit | Every expected historical sample has all required views; no silent intersection selection |
| P0 | Lock near-duplicate policy using historical data only | Prespecified policy, no after-test tuning (§11.2) |
| P1 | Evaluate throughput on representative training file-size strata | End-to-end bytes→stored-image times, sizes and failure counts, not only inner transforms |
| P1 | Check SBSMI all-zero/low-dynamic-range outliers | Descriptive diagnostics on actual stored images; do not filter after peeking at future performance |
| P1 | Integrate three-view PyTorch loaders into the planned branch models | Each batch joins identical SHA/label/chronological split; no extra normalization |
| P1 | Write config-frozen protocol record | Freeze preprocessors, encoders, checkpoint rules, temperatures and reliability weights (§7) |
| Later | Process Apr–Sep and evaluate frozen six monthly sets | No tuning after future outcomes; keep §12.5 adaptation extension separate |

### Recommendations on reporting

- **Acceptable statement now:** “In a 10-sample train-period BODMAS engineering test, the optimized V2 SBSMI and Entropy implementations reproduced reference V2 stored images, with aggregate transform-time ratios of 57.88× and 5.93×, respectively.” This is supported by the **user-uploaded comparison CSV**.
- **Do not state:** “Our classifier is more accurate,” “we improved temporal robustness,” or any April–September macro-F1 result; these data are not in the engineering tests.
- Any comparison between legacy **V1 and V2** must explicitly mention the **fractional-area vs PyTorch-area** preprocessing difference and avoid attributing numeric changes purely to implementation speed.
- If future data have already been evaluated under a frozen setup, proposed alterations to a frozen component are **protocol deviations**, not silent incremental improvements. Historical engineering tests are compatible with the protocol only to the extent the configuration is not frozen or the separate extension track is clearly marked (**§7; §12.5**).

## 8. Short handover summary for teammates

**Build one sample manifest, do not build one per laptop.** Use the repo's training-only 51-family mapping and fixed observation-time periods. Run the **same** V2 generator on three authorized machines, with one view per laptop and matching source/configuration fingerprints. Store `train/month/view` and `validation/month/view` outputs. Transfer the complete per-laptop job folders to the coordinator, merge using original SHA and verify that all three correspond to the same disarmed-content SHA. Use `test/month/view` only after the primary frozen configuration is documented.

**Use fast SBSMI/Entropy only as versioned equivalents of reference V2, never as a stealth change to V1.** The uploaded 10-sample real-BODMAS comparison passes array checks and shows substantial isolated-transform speedups, but the team still needs a full train/validation generation audit and genuine end-to-end performance measurements. No primary classification metric can be claimed from today's tests.

---

## Appendix A. Full function index for the superseded initial V2 generator

This inventory is included because those files were **also created in the reporting window**. **Do not install this initial generator in place of the recommended distributed generator.** Its functionality was reorganized across `src/data/multiview_contract.py`, `scripts/generate_multiview_dataset_v2.py`, and `scripts/merge_multiview_dataset.py`.

**Initial `scripts/generate_multiview_dataset.py` functions:**

| Earlier function | Responsibility in the initial prototype | Recommended successor |
|---|---|---|
| `sha256_file` | Digest file contents for provenance | `file_sha256` in shared contract |
| `hash_json` | Produce stable configuration fingerprint | `canonical_hash` |
| `atomic_json` | Save provenance without partial JSON | `write_json_atomic` |
| `load_csv` | Parse CSV input | `read_csv` |
| `load_mapping` | Enforce frozen family/class mapping | `load_mapping` in shared contract |
| `_sha` | Normalize and check SHA identifier | `load_manifest` checks |
| `_family` | Normalize/check family string | `load_manifest`/`load_mapping` |
| `_canonical_split` | Normalize split name | `normalize_split` |
| `load_manifest_pair` | Join/validate historical manifest with ZIP index; enforce split/date and class mappings | `build_temporal_manifest` plus `load_manifest` / optional `--index` |
| `index_source` | Enumerate SHA-named source binaries | `build_source_index` |
| `select_source_paths` | Match manifest records to ZIP/folder entries | `select_sources` |
| `open_sample` | Open one ZIP member/extracted file | `open_source` |
| `DigestReader` | Digest and count bytes processed by a view | New generator's `DigestReader` |
| `verify_input_fingerprint` | Compare observed bytes to expected source fingerprint | `verified_source` |
| `output_paths` | Construct split-specific image and report paths | `output_paths` in shared contract (now includes month) |
| `validate_array` | Verify image dimensions/dtype/range | `validate_image` |
| `save_array_atomic` | Safely store PNG/NPY | `save_image_atomic` |
| `load_output` | Decode saved image and check format | `load_image` |
| `write_preview` | Generate optional 8-bit visual preview | `--save-previews` in new generator |
| `expected_report` | Define per-output provenance expectations | `expected_report` in contract |
| `valid_cache` | Detect missing/damaged/mismatched output | `valid_report` + `cached_or_none` |
| `generate_one` | Calculate and save one selected view | `make_image` plus the new `run` loop |
| `configuration` | Record algorithms, source and manifest hashes | `build_config` |
| `ensure_configuration` | Prevent incompatible reuse of output root | `ensure_output_config` |
| `build_parser` | Define CLI options | `parse_args` |
| `run` | Main generator flow | distributed generator `run` |
| `main` | CLI wrapper | distributed generator `main` |

The initial `scripts/audit_multiview_dataset.py` had `parse_args`, `audit`, and `main`. Its earlier `--output-dir`-style interface was superseded by the distributed auditor's `--root` interface. The initial `src/preprocessing/sbsmi_stream_v2.py` had `stream_to_sbsmi_exact`, a separate integer-quantized transition generator; it is **not needed** when retaining the group's tested `implementation_sbsmi.py` plus `sbsmi_vectorized.py`. Reference entropy/raw streamers in that initial patch duplicate the modules explained in §4.7–4.10. The initial `tests/test_generate_multiview_dataset_v2.py` covered split separation, ZIP/folder parity, future guard, invalid dates, missing sources, per-split limits, chunk invariance, fractional-area reference calculation, V1/V2 resize differences, tail behavior, previews and audit correctness. Those checks motivated the later distributed suite.

### Appendix B. Existing `sbsmi.py` consistency warning

The repository contains **two** SBSMI-related implementation paths. `implementation_sbsmi.stream_to_sbsmi` (used by the existing group generator) quantizes directly from float64 probabilities using `floor(probabilities * 255)`. The separate `src/preprocessing/sbsmi.py` helper casts a scaled probability image to **float32 before flooring**, which can theoretically change a value near an integer boundary. The equivalence tests in this report compare **against `implementation_sbsmi.py`**, not that other helper. Before a future research freeze, explicitly document the active production path and confirm the MSB-first and final-partial-state conventions against the paper/source notes. Do not silently make a representation change after freezing.

### Appendix C. Reproducibility and privacy record

Keep a local **non-public** engineering archive containing: the exact scripts and Git commit hashes; a copy/hash of the final manifest and 51-family mapping; per-view job configs and logs; full merge/audit reports; `comparison.csv`; the two contact sheets; and, if permitted, per-output content hashes. Add to `.gitignore` at minimum:

```gitignore
# Private generated representations, reports and large files
data/derived/
data/manifests/multiview_v2_manifest.csv
```

Do not commit or redistribute restricted disarmed binaries or their binary-derived representations without verifying the dataset owner's conditions. **Raw-byte images in particular may retain information about the original binary**, so treat them as sensitive research data. Public paper claims should rely on the documented, protocol-compliant test evaluation, not on unverified progress reports or visual observations.
