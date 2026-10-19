# TRAMIF chronological training

The training code uses the existing `TemporalMultiViewDataset` and
`MalwareImageBranch` contract. It does not modify the independent MalCSBSV
10-fold reproduction (`src/training/reproduction.py`).

## What is implemented

- One independent classifier for each selected view and seed. The committed
  `MalSBSLCNet` class is the only CNN implemented in the current source and
  is used for all three views in the example YAML. It is **not evidence** that
  three view-specific architectures have been implemented or validated.
- 51 locked classes from `bodmas_eligible_families_v0.csv`.
- Native V2 arrays, with **no additional resizing or normalization**:
  `raw_byte` and `entropy` are `[64,64]` float32 `.npy` in `[0,1]`;
  `sbsmi` is 8-bit grayscale `.png` converted to `[0,1]`.
- Pilot fit: 2019-08 to 2019-11; pilot holdout: 2019-12 and 2020-01.
  The *pilot* model is only for choosing one number of epochs per view.
- Final fit: all six months 2019-08 to 2020-01. Never train or choose
  checkpoints using 2020-02 to 2020-09.
- The final checkpoint exports the uncalibrated model with both logits and
  embeddings available when it is loaded. Calibration, drift, reliability
  and fusion are not implemented by this training command.

## Requirements

From the repository root, with the project's Python environment activated:

```powershell
python -m pip install torch numpy Pillow scikit-learn PyYAML pytest
Copy-Item configs/train_temporal.example.yaml configs/train_temporal.yaml
```

For PyTorch with GPU support, use the install command recommended for your
hardware by PyTorch. The code was smoke-tested on CPU, not benchmarked on GPU.

In `configs/train_temporal.yaml`, set `dataset.root` to the V2 generated/merged
image folder containing `config.json` and the `train/YYYY-MM/<view>/` images.
Leave `dataset.manifest` and `dataset.mapping` pointing to the authoritative
V2 manifest and family map. You must have the actual generated image files;
the GitHub ZIP only includes the CSV manifest, not the private binary-derived
images. Manifest hashes in `config.json` must match those CSVs exactly.

**Important:** use `python -m scripts.train_temporal` instead of
`python scripts/train_temporal.py`; this repo imports from `src` and needs the
repository root on Python's module search path.

## Smoke test / preflight

```powershell
python -m pytest -q tests/training/test_temporal.py
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase pilot --view sbsmi --seeds 11 --dry-run
```

`--dry-run` verifies CSV validity, chronology, mapping and preprocessing hashes,
but it does **not** check whether all image files are present. Actual training
fails if a required image is missing. Inspect the dataset first with
`python -m scripts.audit_multiview_dataset --help`.

Start a one-seed pilot to check the model output, VRAM and dataloader:

```powershell
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase pilot --view sbsmi --seeds 11 --skip-completed
```

The following is the *actual* five-seed workflow. All three modalities can be
trained on separate machines using the same revision, YAML settings,
manifest/mapping files and preprocessing config ID:

```powershell
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase pilot --view sbsmi --seeds 11,22,33,44,55 --skip-completed
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase pilot --view raw_byte --seeds 11,22,33,44,55 --skip-completed
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase pilot --view entropy --seeds 11,22,33,44,55 --skip-completed
```

Bring the three `runs/temporal/<view>/pilot/` folders together. Then:

```powershell
python -m scripts.select_temporal_epochs --output-root runs/temporal --seeds 11,22,33,44,55 --save runs/temporal/epoch_plan.json
```

The script averages per-epoch **mean monthly macro-F1 over 51 fixed classes**
across the five pilot seeds, chooses the highest mean and breaks ties in favor
of the earliest epoch. Both pilot months must be nonempty. Missing families in
a month count as zero F1 under this convention; document this in the paper.

Final training (each command independently executable on one laptop):

```powershell
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase final --view sbsmi --seeds 11,22,33,44,55 --epoch-plan runs/temporal/epoch_plan.json --skip-completed
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase final --view raw_byte --seeds 11,22,33,44,55 --epoch-plan runs/temporal/epoch_plan.json --skip-completed
python -m scripts.train_temporal --config configs/train_temporal.yaml --phase final --view entropy --seeds 11,22,33,44,55 --epoch-plan runs/temporal/epoch_plan.json --skip-completed
```

The plan checks image preprocessing, manifest, mapping, model class and train
settings. Pilot and final run *separate freshly initialized models*; a pilot
checkpoint is not reused as the starting weights for final training.

## Checkpoint and resume

Each seed has its own directory:

```text
runs/temporal/<view>/pilot/seed_11/
  metadata.json
  history.jsonl
  last.pt
  summary.json
runs/temporal/<view>/final/seed_11/
  metadata.json
  history.jsonl
  last.pt
  final.pt
  summary.json
```

`last.pt` is an epoch-boundary resume checkpoint containing weights, AdamW,
cosine scheduler, AMP scaler, RNG state and history. Rerun the same command
with `--skip-completed`: incomplete local runs resume; completed runs skip.
Resume at mid-epoch starts again from the previous completed epoch boundary.
**Only load trusted checkpoints**: `last.pt` is a pickle-based PyTorch file.

`final.pt` is a smaller inference checkpoint, containing `model` state_dict
and `metadata`. Its logits are *uncalibrated*. Do not apply softmax before
later temperature scaling. A final result is one model per view **per seed**;
15 checkpoints across three views and five seeds are not a 15-model ensemble.

## Limitations / protocol decisions

1. No private V2 PNG/NPY image arrays were included in the uploaded ZIP, so
   the full 18,061-sample training run could not be benchmarked here.
2. This is not a deployment/fusion script. Dedicated per-view CNNs must be
   supplied as new `MalwareImageBranch` classes if required by the study.
3. Fixed-family macro F1 is deliberately strict for months lacking a class.
4. Multi-worker throughput and CUDA determinism should be validated on the
   training GPUs. Start with `num_workers: 0` on Windows, benchmark 2 and 4;
   pin memory is used automatically on CUDA.
5. No run should begin without notifying the group and recording the frozen
   experiment configuration and machine assignment.
6. The 10-fold MalCSBSV **reproduction** is a separate experiment. Do not
   merge its fold metrics or checkpoints with this chronological study.
