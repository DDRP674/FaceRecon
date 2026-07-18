from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import PARTITION_NAMES, Paths


@dataclass(frozen=True)
class CelebARecord:
    image_id: str
    image_path: Path
    partition: int
    split: str
    identity: int | None = None


def find_identity_file(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "identity_CelebA.txt",
        data_dir / "identity_CelebA.csv",
        data_dir / "identity_celeba.txt",
        data_dir / "identity_celeba.csv",
        data_dir / "list_identity_celeba.csv",
        data_dir / "list_identity_celeba.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_partitions(paths: Paths) -> dict[str, int]:
    if not paths.partition_csv.exists():
        raise FileNotFoundError(f"Official CelebA split file not found: {paths.partition_csv}")
    with paths.partition_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row["image_id"]: int(row["partition"]) for row in reader}


def read_identities(data_dir: Path) -> dict[str, int]:
    path = find_identity_file(data_dir)
    if path is None:
        raise FileNotFoundError(
            "CelebA identity labels are required for identity retrieval/ArcFace training. "
            "Put identity_CelebA.txt or identity_CelebA.csv under data/celeba/."
        )

    identities: dict[str, int] = {}
    with path.open(newline="") as f:
        sample = f.readline()
        f.seek(0)
        if "," in sample:
            reader = csv.DictReader(f)
            image_key = "image_id" if "image_id" in (reader.fieldnames or []) else (reader.fieldnames or [])[0]
            id_key = "identity" if "identity" in (reader.fieldnames or []) else (reader.fieldnames or [])[1]
            for row in reader:
                identities[row[image_key]] = int(row[id_key])
        else:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].lower() != "image_id":
                    identities[parts[0]] = int(parts[1])
    return identities


def load_records(paths: Paths, require_identity: bool = False, check_exists: bool = False) -> list[CelebARecord]:
    partitions = read_partitions(paths)
    identities = read_identities(paths.data_dir) if require_identity else {}
    records: list[CelebARecord] = []
    for image_id, partition in partitions.items():
        image_path = paths.image_dir / image_id
        if not image_path.exists():
            continue
        identity = identities.get(image_id)
        if require_identity and identity is None:
            continue
        if check_exists and not image_path.exists():
            continue
        records.append(
            CelebARecord(
                image_id=image_id,
                image_path=image_path,
                partition=partition,
                split=PARTITION_NAMES[partition],
                identity=identity,
            )
        )
    return records


def write_split_manifest(paths: Paths, out_csv: Path, require_identity: bool = False) -> None:
    records = load_records(paths, require_identity=require_identity, check_exists=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["image_id", "image_path", "partition", "split", "identity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "image_id": r.image_id,
                    "image_path": str(r.image_path),
                    "partition": r.partition,
                    "split": r.split,
                    "identity": "" if r.identity is None else r.identity,
                }
            )
