# TRAMIF training integration audit — 2026-10-15

## Verified repository contracts

Reviewed the actual uploaded project archive (not only the earlier standalone
training patch). The chronological code uses:

- `data/manifests/multiview_v2_manifest.csv`: 47,157 sample metadata rows;
  18,061 train, 8,263 validation and 20,833 future test.
- `data/manifests/bodmas_eligible_families_v0.csv`: 51 fixed family classes.
- `src/data/temporal_multiview_dataset.py`: loads `raw_byte`, `entropy`, `sbsmi`
  as single-channel 64×64 tensors. The underlying preprocessing contract keeps
  the two first types as float32 NPY and SBSMI as uint8 PNG. Training inputs
  are already normalized into `[0,1]` in the loader.
- `src/models/malsbslcnet.py`: implemented CNN with `BranchOutput(logits, embedding)`.
- `PROTOCOL.md`: train 2019-08–2020-01, validation 2020-02–2020-03 and
  future test 2020-04–2020-09.
- `src/training/reproduction.py`: independent 10-fold MalCSBSV reproduction.

The previous V2 patch using generic image paths and an ordinary logits tensor
was not directly compatible with these actual interfaces. This integration
uses the repository-native dataset and model instead.

## New files

- `src/training/temporal.py`: training engine, pilot temporal holdout, model
  factory, seed management, gradient accumulation, AMP, checkpoint/resume.
- `scripts/train_temporal.py`: CLI and strict protocol/hash preflight.
- `scripts/select_temporal_epochs.py`: fixed epoch selection across pilot seeds.
- `configs/train_temporal.example.yaml`: view-specific CNN configuration.
- `tests/training/test_temporal.py`: seven targeted tests using synthetic V2
  64×64 images and the actual 51-class model.
- `docs/TRAMIF_TEMPORAL_TRAINING.md`: usage guide.

Only `.gitignore` was appended to ignore the local editable YAML config.
All existing reproduction and preprocessing modules were left unchanged.

## Verification

- Full repository suite including the new tests: **354 passed**, 13 warnings,
  in CPU environment.
- Dedicated new tests: **7 passed**.
- End-to-end CLI smoke: `--dry-run` → pilot → epoch selection → final completed
  successfully on synthetic V2 data; output `final.pt` was produced.
- Reproducibility: on CPU synthetic input, interrupted training resumed from
  `last.pt` produced identical model tensors to uninterrupted training.
- **Not verified:** actual image files or full-scale training speed/GPU memory.
  Private V2 images are absent from the uploaded archive, so no malware
  classification accuracy or real-data performance claim can be made.

## Before the group trains

1. Confirm the dataset root has the V2 arrays and correct `config.json`, and
   choose which CNN architecture each view should actually use. The repo
   currently contains `MalSBSLCNet` but does not implement three independent
   view-specific network classes.
2. Document the agreed optimizer, learning rate, epochs, augmentation policy,
   seed set and fixed-class macro-F1 convention. YAML defaults are a starting
   configuration, **not independently optimized hyperparameters**.
3. Run pilot with one seed/view first on the real GPU, then all five seeds.
4. Notify the group before full training; distribute matching manifests,
   mapping, source revision and epoch plan.
5. Keep February–September out of encoder training and epoch selection.
   Apply the protocol's separate calibration/reliability work later.
