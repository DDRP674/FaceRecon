from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data" / "celeba"
    work_dir: Path = REPO_ROOT / "work"
    model_dir: Path = REPO_ROOT / "models"

    @property
    def image_dir(self) -> Path:
        nested = self.data_dir / "img_align_celeba" / "img_align_celeba"
        if nested.exists():
            return nested
        return self.data_dir / "img_align_celeba"

    @property
    def partition_csv(self) -> Path:
        return self.data_dir / "list_eval_partition.csv"

    @property
    def embeddings_dir(self) -> Path:
        return self.work_dir / "embeddings"

    @property
    def latents_dir(self) -> Path:
        return self.work_dir / "latents"

    @property
    def generated_dir(self) -> Path:
        return self.work_dir / "generated"

    @property
    def metrics_dir(self) -> Path:
        return self.work_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.work_dir / "checkpoints"

    @property
    def realvisxl_path(self) -> Path:
        return self.model_dir / "realvisxl-v3.0"

    @property
    def ip_adapter_ckpt(self) -> Path:
        return self.model_dir / "ip-adapter-faceid" / "ip-adapter-faceid_sdxl.bin"


NOISE_LEVELS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
PARTITION_NAMES = {0: "train", 1: "val", 2: "test"}

