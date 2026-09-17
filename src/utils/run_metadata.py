from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess


@dataclass
class RunMetadata:
    run_name: str
    seed: int
    config_name: str
    preprocessing_version: str
    dataset_manifest: str
    git_commit: str
    created_at_utc: str


def get_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()


def create_run_metadata(
    run_name: str,
    seed: int,
    config_name: str,
    preprocessing_version: str,
    dataset_manifest: str,
) -> RunMetadata:
    return RunMetadata(
        run_name=run_name,
        seed=seed,
        config_name=config_name,
        preprocessing_version=preprocessing_version,
        dataset_manifest=dataset_manifest,
        git_commit=get_git_commit(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_run_metadata(metadata: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)