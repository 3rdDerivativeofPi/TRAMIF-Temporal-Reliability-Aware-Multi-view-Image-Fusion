"""CPU integration checks for the TRAMIF V2 training contract."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.select_temporal_epochs import choose_epochs
from src.data.multiview_contract import file_sha256, write_json_atomic
from src.training.temporal import (SafeShuffleBatchSampler, _branch_logits,
                                   create_branch, make_datasets, run_seed)


def _fixture(tmp_path: Path):
    mapping = Path(__file__).parents[2] / "data/manifests/bodmas_eligible_families_v0.csv"
    manifest = tmp_path / "manifest.csv"
    labels = [row for row in csv.DictReader(mapping.open(encoding="utf-8"))]
    folders = tmp_path / "images"
    months = ["2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01"]
    rows = []
    for i, month in enumerate(months):
        for j in range(4):
            sha = f"{i * 4 + j + 1:064x}"
            rows.append({"sha": sha, "family": labels[j % 2]["family"],
                         "class_id": labels[j % 2]["class_id"], "split": "train",
                         "month": month, "timestamp": f"{month}-10T02:00:00+00:00",
                         "file_size": "", "zip_crc32": ""})
            for view in ("raw_byte", "entropy", "sbsmi"):
                dest = folders / "train" / month / view
                dest.mkdir(parents=True, exist_ok=True)
                pixels = np.full((64, 64), (i + 1) * 10 + j, dtype=np.uint8)
                if view == "sbsmi":
                    Image.fromarray(pixels, mode="L").save(dest / f"{sha}.png")
                else:
                    np.save(dest / f"{sha}.npy", pixels.astype("float32") / 255)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(folders / "config.json", {
        "manifest_sha256": file_sha256(manifest),
        "mapping_sha256": file_sha256(mapping),
        "config_id": "synthetic-v2-test",
    })
    cfg = {
        "dataset": {"root": str(folders), "manifest": str(manifest), "mapping": str(mapping)},
        "models": {view: {"class_path": "src.models.malsbslcnet:MalSBSLCNet", "kwargs": {}}
                   for view in ("raw_byte", "entropy", "sbsmi")},
        "training": {"epochs": 2, "batch_size": 4, "num_workers": 0,
                     "learning_rate": 1e-4, "weight_decay": 1e-6,
                     "grad_accumulation": 2, "grad_clip": 1.0, "amp": False,
                     "deterministic": True},
    }
    return cfg


def test_sampler_repeatable_and_no_singleton():
    a = SafeShuffleBatchSampler(9, 4, 17)
    batches = list(a)
    assert len(batches) == len(a) == 2
    assert sorted(sum(batches, [])) == list(range(9))
    assert min(map(len, batches)) >= 2
    assert list(a) == batches
    a.epoch = 1
    assert list(a) != batches
    with pytest.raises(ValueError):
        SafeShuffleBatchSampler(9, 1, 17)


def test_model_interface_real_cnn():
    torch.set_num_threads(1)
    model = create_branch({"class_path": "src.models.malsbslcnet:MalSBSLCNet"}, 51).eval()
    with torch.inference_mode():
        batch = torch.zeros(2, 1, 64, 64)
        output = model(batch)
        assert _branch_logits(output, torch.zeros(2, dtype=torch.long), 51).shape == (2, 51)
        assert output.embedding.shape == (2, 2048)


def test_dataset_uses_real_v2_contract(tmp_path):
    cfg = _fixture(tmp_path)
    d, h = make_datasets(Path(cfg["dataset"]["root"]), Path(cfg["dataset"]["manifest"]),
                         Path(cfg["dataset"]["mapping"]), "entropy", "pilot")
    assert len(d) == 16 and len(h) == 8
    x, y = d[0]
    assert x["entropy"].shape == (1, 64, 64)
    assert 0 <= float(x["entropy"].min()) <= float(x["entropy"].max()) <= 1
    assert 0 <= y <= 50
    assert h[0][2] in ("2019-12", "2020-01")
    assert len(make_datasets(Path(cfg["dataset"]["root"]),
                Path(cfg["dataset"]["manifest"]), Path(cfg["dataset"]["mapping"]),
                "sbsmi", "final")[0]) == 24


def test_pilot_final_epoch_plan_and_skip(tmp_path):
    torch.set_num_threads(1)
    cfg = _fixture(tmp_path)
    output = tmp_path / "runs"
    for seed in (11, 22):
        summary = run_seed(cfg=cfg, view="raw_byte", seed=seed, phase="pilot", epochs=2,
                           output_root=output, device=torch.device("cpu"))
        assert summary["samples_fit"] == 16
        assert summary["samples_validation"] == 8
        assert run_seed(cfg=cfg, view="raw_byte", seed=seed, phase="pilot", epochs=2,
                        output_root=output, device=torch.device("cpu"), skip_completed=True) == summary
    plan = choose_epochs(output, ("raw_byte",), [11, 22], "mean_month_macro_f1")
    assert 1 <= plan["views"]["raw_byte"]["selected_epoch"] <= 2
    assert plan["views"]["raw_byte"]["seeds"] == [11, 22]
    summary = run_seed(cfg=cfg, view="raw_byte", seed=11, phase="final", epochs=1,
                       output_root=output, device=torch.device("cpu"))
    saved = torch.load(output / "raw_byte/final/seed_11/final.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"]["num_classes"] == 51
    assert summary["samples_validation"] == 0
    assert saved["model"]


def test_resume_matches_uninterrupted_cpu(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = _fixture(tmp_path)
    from src.training import temporal
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    run_seed(cfg=cfg, view="sbsmi", seed=31, phase="final", epochs=2,
             output_root=clean, device=torch.device("cpu"))
    original = temporal._train_epoch

    def interrupt(*args, **kwargs):
        if args[-1] == 1:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(temporal, "_train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_seed(cfg=cfg, view="sbsmi", seed=31, phase="final", epochs=2,
                 output_root=resumed, device=torch.device("cpu"))
    monkeypatch.setattr(temporal, "_train_epoch", original)
    run_seed(cfg=cfg, view="sbsmi", seed=31, phase="final", epochs=2,
             output_root=resumed, device=torch.device("cpu"))
    a = torch.load(clean / "sbsmi/final/seed_31/final.pt", weights_only=True)["model"]
    b = torch.load(resumed / "sbsmi/final/seed_31/final.pt", weights_only=True)["model"]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert len((resumed / "sbsmi/final/seed_31/history.jsonl").read_text().splitlines()) == 2


def test_epoch_plan_rejects_incomplete_or_changed_seed(tmp_path):
    cfg = _fixture(tmp_path)
    output = tmp_path / "runs"
    with pytest.raises(FileNotFoundError):
        choose_epochs(output, ("raw_byte",), [11], "mean_month_macro_f1")
    root = output / "raw_byte/pilot/seed_11"
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps({"completed": False, "phase": "pilot", "seed": 11}))
    (root / "metadata.json").write_text(json.dumps({"phase": "pilot", "seed": 11}))
    (root / "history.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Incomplete"):
        choose_epochs(output, ("raw_byte",), [11], "mean_month_macro_f1")


def test_missing_image_fails(tmp_path):
    cfg = _fixture(tmp_path)
    next((Path(cfg["dataset"]["root"]) / "train").rglob("raw_byte/*.npy")).unlink()
    with pytest.raises(FileNotFoundError):
        make_datasets(Path(cfg["dataset"]["root"]), Path(cfg["dataset"]["manifest"]),
                      Path(cfg["dataset"]["mapping"]), "raw_byte", "final")
