TRAMIF chronological training patch

Add the files in this archive into the same relative directories of the original TRAMIF repository. Do not overwrite existing training/reproduction.py or existing models. From the repo root run: python -m pytest -q tests/training/test_temporal.py. See docs/TRAMIF_TEMPORAL_TRAINING.md. Add configs/train_temporal.yaml to .gitignore when copying the example config.
