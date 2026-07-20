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


IDENTITY_COLUMN_NAMES = {"identity", "identity_id", "person_id", "subject_id", "class_id", "label"}
IMAGE_COLUMN_NAMES = {"image_id", "image", "filename", "file", "path"}


def _looks_like_identity_csv(path: Path) -> bool:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            fields = {field.strip().lower() for field in (reader.fieldnames or [])}
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return bool(fields & IMAGE_COLUMN_NAMES) and bool(fields & IDENTITY_COLUMN_NAMES)


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
    for path in sorted(data_dir.glob("*.csv")):
        if _looks_like_identity_csv(path):
            return path
    return None


def describe_available_label_files(data_dir: Path) -> str:
    lines = []
    for path in sorted(data_dir.glob("*.csv")):
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
        except (OSError, UnicodeDecodeError, csv.Error):
            fields = []
        preview = ", ".join(fields[:8])
        if len(fields) > 8:
            preview += ", ..."
        lines.append(f"{path.name}: {preview}")
    return "; ".join(lines) if lines else "no CSV files found"


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
            "The CSV files currently under data/celeba do not contain an identity/person_id column. "
            f"Found: {describe_available_label_files(data_dir)}. "
            "Put identity_CelebA.txt or a CSV with image_id and identity columns under data/celeba/."
        )

    identities: dict[str, int] = {}
    with path.open(newline="") as f:
        sample = f.readline()
        f.seek(0)
        if "," in sample:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            lower_to_field = {field.strip().lower(): field for field in fieldnames}
            image_key = next((lower_to_field[name] for name in IMAGE_COLUMN_NAMES if name in lower_to_field), fieldnames[0])
            id_key = next((lower_to_field[name] for name in IDENTITY_COLUMN_NAMES if name in lower_to_field), fieldnames[1])
            for row in reader:
                identities[row[image_key]] = int(row[id_key])
        else:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].lower() != "image_id":
                    identities[parts[0]] = int(parts[1])
    return identities


def read_landmarks(data_dir: Path) -> dict[str, list[list[float]]]:
    path = data_dir / "list_landmarks_align_celeba.csv"
    if not path.exists():
        raise FileNotFoundError(f"CelebA aligned landmark file not found: {path}")
    out: dict[str, list[list[float]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["image_id"]] = [
                [float(row["lefteye_x"]), float(row["lefteye_y"])],
                [float(row["righteye_x"]), float(row["righteye_y"])],
                [float(row["nose_x"]), float(row["nose_y"])],
                [float(row["leftmouth_x"]), float(row["leftmouth_y"])],
                [float(row["rightmouth_x"]), float(row["rightmouth_y"])],
            ]
    return out


def load_records(paths: Paths, require_identity: bool = False, check_exists: bool = False) -> list[CelebARecord]:
    partitions = read_partitions(paths)
    identities = read_identities(paths.data_dir) if require_identity else {}
    records: list[CelebARecord] = []
    for image_id, partition in partitions.items():
        image_path = paths.image_dir / image_id
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
