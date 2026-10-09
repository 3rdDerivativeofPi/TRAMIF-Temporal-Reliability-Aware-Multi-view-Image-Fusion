# TRAMIF V2: distributed multi-view dataset generator

**Status:** Proposed source-code patch; not merged to GitHub and **not run on private BODMAS**. Tested on synthetic fixtures (see tests). Applied against the group's `main` as inspected 2026-10-10 (commit `ecb0bef`).

## 0. Safety and scientific protocol

- Family eligibility/class indices MUST come from the **training-derived** `data/manifests/bodmas_eligible_families_v0.csv` (51 classes; never reconstruct mapping from the whole metadata).
- **Train:** 2019-08 through 2020-01. **Validation:** 2020-02 through 2020-03. **Test:** 2020-04 through 2020-09. The manifest builder uses BODMAS *observation timestamps*, not PE timestamps.
- **Before using April–September for evaluation**, freeze the preprocessing, encoders, checkpoint choices, temperatures and reliability rule (§7). The generator guards test image construction with `--frozen-config-id`, but a command-line guard cannot substitute for a documented research freeze.
- This is **V2**: Raw-byte/Entropy use explicit fractional geometric-area resizing. Original V1 used PyTorch `mode="area"`; they are *not generally pixel-identical*. Existing V1-trained checkpoints must not silently switch to V2 inputs (framework §4.4).
- Keep your existing SBSMI generation / 10-fold MalCSBSV reproduction tests independent of chronological TRAMIF experiments.
- Do not commit or publicly redistribute restricted binaries, binary-derived representations, or sensitive reports unless the BODMAS owner's conditions permit it. Raw-byte views may preserve sensitive byte information.

## 1. Files to copy into the group repository

| File in patch | Action |
|---|---|
| `scripts/build_temporal_manifest.py` | New: creates one shared canonical manifest from BODMAS metadata and locked family CSV |
| `scripts/generate_multiview_dataset_v2.py` | New: same generator on all laptops; **does not replace** V1 script |
| `scripts/merge_multiview_dataset.py` | New: verifies and merges modality outputs from different laptops |
| `scripts/audit_multiview_dataset.py` | New: audits merged or single-laptop outputs |
| `src/data/multiview_contract.py` | New: sample identities, splits, hashes, shapes, reports |
| `src/data/temporal_multiview_dataset.py` | New: chronological PyTorch dataset; separate from CV reproduction |
| `src/preprocessing/raw_byte_stream.py` | New: streaming geometric-area Raw-byte |
| `src/preprocessing/entropy_core.py` | New: Shannon entropy definition |
| `src/preprocessing/entropy_stream.py` | New: overlapping streaming entropy |
| `src/preprocessing/entropy_image_stream.py` | New: streaming geometric-area Entropy images |
| `tests/test_distributed_multiview_v2.py` | New: synthetic integration tests |

**Required existing GitHub file (do NOT replace):** `src/preprocessing/implementation_sbsmi.py` with `stream_to_sbsmi`.

Dependencies: `numpy`, `pandas`, `Pillow`, `tqdm`, `torch`; `pytest` for tests. The script supports Python 3.10+ with the already installed project dependencies.

Apply on a feature branch. V1 is left intact to retain previous experiment provenance:

```powershell
git switch main
git pull
git switch -c feature/distributed-multiview-v2
# Extract patch ZIP and copy its scripts/, src/, tests/, docs/ into matching repo paths.
python -m pytest -q tests/test_distributed_multiview_v2.py
```

The test suite requires the **existing** `src/preprocessing/implementation_sbsmi.py` in the repo. Use only synthetic fixtures in these tests.

## 2. Canonical manifest: create ONCE, then distribute unchanged

Run on the coordinator's machine from the repository root (PowerShell):

```powershell
python -m scripts.build_temporal_manifest `
  --metadata "E:\BODMAS_GW\bodmas_metadata.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output "data\manifests\multiview_v2_manifest.csv"
```

Alternative metadata input: `data\manifests\bodmas_family_v0.csv` when generated locally by the group audit script. The builder automatically keeps only the 51 historically eligible families and valid protocol dates. It outputs columns:

`sha,family,class_id,split,month,timestamp,file_size,zip_crc32`

`sha` is the original sample ID, **not** the disarmed binary SHA. `file_size` and `zip_crc32` may be blank in this canonical manifest; when present, the generator checks them against the input archive/file.

Optional `--index` accepts a previously audited CSV with `sha,file_size,zip_crc32` to fill known fingerprints.

**Copy the exact same manifest CSV and class-map CSV bytes to all three laptops**. Verify hashes:

```powershell
Get-FileHash "data\manifests\multiview_v2_manifest.csv" -Algorithm SHA256
Get-FileHash "data\manifests\bodmas_eligible_families_v0.csv" -Algorithm SHA256
```

The builder may include future-test rows so the *evaluation cohort* is defined consistently; such rows and their labels **must not be used for model development, coefficient selection or preprocessing optimization**. The generator defaults to training and validation and requires an explicit frozen configuration for test processing.

## 3. Three laptops, one script, three modalities

The commands below assume each member has an authorized local copy of the approved disarmed ZIP. The `--output-dir` is local to that laptop. The other two modalities are not produced on that machine.

### Laptop A — SBSMI

```powershell
python -m scripts.generate_multiview_dataset_v2 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "E:\TRAMIF_jobs\sbsmi" `
  --modalities sbsmi --splits train validation `
  --worker-id laptop-A
```

### Laptop B — Raw-byte

```powershell
python -m scripts.generate_multiview_dataset_v2 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "E:\TRAMIF_jobs\raw" `
  --modalities raw_byte --splits train validation `
  --worker-id laptop-B
```

### Laptop C — Entropy

```powershell
python -m scripts.generate_multiview_dataset_v2 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "E:\TRAMIF_jobs\entropy" `
  --modalities entropy --splits train validation `
  --worker-id laptop-C
```

**Smoke test:** first add `--dry-run`, then run again without `--dry-run` but with `--limit 5` (5 samples *per selected split* on that worker). After the smoke test passes, remove `--limit 5`; verified files are reused.

**ZIP or folder:** replace `--source` with `E:\BODMAS_GW\altered` for extracted files. For the generator, output identity depends on the selected original SHA and actual disarmed bytes, not the source folder location.

**Run only one modality:** `--modalities sbsmi`, `raw_byte`, or `entropy`. For processing all modalities on one machine: `--modalities all`.

**Extra parallelism within a modality:** for two workers computing different SHA partitions, use `--num-shards 2 --shard-id 0` and `--num-shards 2 --shard-id 1`. Assignments are derived from integer SHA modulo shard count, independent of CSV order. When transferring to the coordinator, include both output roots for that modality.

**Resume:** run the same command again. It verifies output checksums and stored provenance, and rebuilds missing/corrupt images. To additionally rehash original source bytes on cache hits, add `--verify-source-on-skip` (slower). To force a rebuild add `--overwrite` (same V2 configuration only).

## 4. Transfer and merge

Copy each laptop's **entire output directory**, including `config.json` and `reports/`, to the coordinator. Do not merge by zip member order, image count, filenames alone, or train/validation file discovery. Treat the local output roots as immutable until merge is audited.

For a 5-sample-per-split smoke test, use `--limit 5` on the merge command too. For full runs omit `--limit`.

```powershell
python -m scripts.merge_multiview_dataset `
  --inputs "E:\TRAMIF_collected\sbsmi" "E:\TRAMIF_collected\raw" "E:\TRAMIF_collected\entropy" `
  --output-dir "E:\TRAMIF_collected\merged" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --splits train validation --dry-run
```

If preflight passes, remove `--dry-run` to atomically copy outputs and reports. If a view is missing, duplicate, corrupt, configured differently, or generated from different disarmed content, merging FAILS rather than dropping or reassigning that sample.

Run the full auditor:

```powershell
python -m scripts.audit_multiview_dataset `
  --root "E:\TRAMIF_collected\merged" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --splits train validation --check-extras
```

The historical cohort was *previously* audited as 18,061 train + 8,263 validation. The actual count is determined by the canonical manifest produced and verified on your local source. **Do not claim completed image counts until the real generation/audit has run.**

## 5. Directory layout

```text
E:/TRAMIF_collected/merged/
  config.json
  merged_manifest.csv
  train/
    2019-08/
      raw_byte/<original_sha>.npy
      entropy/<original_sha>.npy
      sbsmi/<original_sha>.png
    2019-09/ ...
    2019-10/ ...
    2019-11/ ...
    2019-12/ ...
    2020-01/ ...
  validation/
    2020-02/raw_byte/, entropy/, sbsmi/
    2020-03/raw_byte/, entropy/, sbsmi/
  test/
    2020-04/ ...
    2020-05/ ...
    2020-06/ ...
    2020-07/ ...
    2020-08/ ...
    2020-09/ ...
  reports/<split>/<month>/<modality>/<original_sha>.json
```

Raw-byte and Entropy `.npy` contain 64x64 float32 values in `[0,1]`. SBSMI `.png` contains integer `uint8` pixels; divide by 255 on model input. Optional `--save-previews` writes view-only PNG previews for Raw-byte and Entropy under a separate `previews/` tree. **Do not substitute preview PNGs for float32 training arrays.**

## 6. Frozen test generation: only AFTER the research freeze

Before the future test, record the historical `CONFIG ID` reported by the generator and freeze the model selection, temperatures, reliability rule, family mapping and preprocessing per framework §7. On each laptop, use its **same historical output root** (which contains `config.json`).

```powershell
python -m scripts.generate_multiview_dataset_v2 `
  --source "E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip" `
  --manifest "data\manifests\multiview_v2_manifest.csv" `
  --mapping "data\manifests\bodmas_eligible_families_v0.csv" `
  --output-dir "E:\TRAMIF_jobs\raw" `
  --modalities raw_byte --splits test `
  --frozen-config-id "PASTE_EXACT_HISTORICAL_CONFIG_ID"
```

Repeat for SBSMI/Entropy with the same frozen config ID and each modality's root. After all three complete, merge and audit `--splits test` separately, also providing `--frozen-config-id`. **Never use test outcomes to change the encoder, preprocessing, temperatures or fusion weights in the primary track.**

## 7. PyTorch integration

```python
from torch.utils.data import DataLoader
from src.data.temporal_multiview_dataset import TemporalMultiViewDataset

train = TemporalMultiViewDataset(
    root=r"E:\TRAMIF_collected\merged",
    manifest="data/manifests/multiview_v2_manifest.csv",
    mapping="data/manifests/bodmas_eligible_families_v0.csv",
    split="train",
)
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
views, labels = next(iter(loader))
# views['raw_byte'], views['entropy'], views['sbsmi']: [B, 1, 64, 64]
# labels: [B], from the SAME frozen class mapping
```

For one branch use `modalities=('entropy',)`. For one validation month use `split='validation', month='2020-02'`. For frozen testing, use `split='test', month='2020-04', allow_test=True`. This dataset **does not** replace the separate 10-fold reproduction DataModule.

## 8. Recommended repository hygiene

Add to the group's `.gitignore` (do not overwrite the whole existing file):

```gitignore
data/derived/
data/manifests/multiview_v2_manifest.csv
```

Store the manifest and dataset artifacts on authorized shared storage; commit only code, tests, documentation, and non-sensitive aggregate audit summaries, respecting BODMAS restrictions. Do not commit private malware binaries or raw-byte image arrays.

## 9. What has been tested (and what has not)

- Synthetic integration: three separate modality roots, matching source bytes, merge, auditing and PyTorch reading.
- Chunk boundaries across file sizes, ZIP vs folder, resume/corruption, missing files, mismatched source content, duplicate modality candidates, invalid split/month, code-config mismatch and future-test guard.
- The tests use artificial bytes, not real malware. They do not measure F1, drift, latency or model accuracy.
- No private BODMAS run, full-dataset audit, source-owner redistribution review, or GitHub commit has been made here. Do a *real source smoke test* and inspect configuration before full generation.

## 10. Implementation caveats

- The builder intentionally relies on a previously frozen **train-derived family CSV**; it never recomputes the eligible families from future data.
- Precision/resize changed from V1 and is versioned; comparison experiments must use identical preprocessing for each method.
- Each laptop can use a different file location and chunk size. Source-code/configuration fingerprints, input SHA/CRC, and the final merge audit protect against inconsistent preprocessing and mismatched bytes.
- The merger is strict. A partial input job or missing modality is an error. Finish/resume jobs first; do not silently train on only fully covered intersection unless that subset is separately defined and reported.
- Cache validation includes output hash and report; `--verify-source-on-skip` adds full source rehash (recommended for final audit after transfer where bandwidth permits).
- Whole-file entropy can be CPU intensive even with bounded memory. Measure actual performance before setting an overnight schedule.

- An audit covering multiple chronological splits also **warns** when different original sample IDs resolve to exactly identical disarmed-content hashes across splits. It does not silently remove those samples; the team must apply its prespecified exact-/near-duplicate policy (§11.2).
