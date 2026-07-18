from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .celeba import CelebARecord
from .config import NOISE_LEVELS, Paths
from .noise import add_gaussian_noise_bgr
from .progress import progress_bar


def load_face_app(ctx_id: int = 0, det_size: tuple[int, int] = (640, 640)):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def embed_image_bgr(app, image_bgr: np.ndarray) -> np.ndarray | None:
    face = largest_face(app.get(image_bgr))
    if face is None:
        return None
    return np.asarray(face.normed_embedding, dtype=np.float32)


def _shard_path(out_dir: Path, level: int, shard_idx: int) -> Path:
    return out_dir / f"noise_{level:02d}" / f"shard_{shard_idx:05d}.pt"


def _atomic_torch_save(payload: dict[str, object], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def _load_partial_shard(out: Path) -> dict[str, object] | None:
    if not out.exists():
        return None
    try:
        return torch.load(out, map_location="cpu")
    except Exception:
        return None


def _is_complete_shard(out: Path, expected_len: int) -> bool:
    payload = _load_partial_shard(out)
    if payload is None:
        return False
    image_ids = payload.get("image_ids", [])
    if len(image_ids) != expected_len:
        return False
    return bool(payload.get("complete", True))


def _payload_to_disk(payload: dict[str, object], out: Path, complete: bool) -> None:
    emb_list = payload["embeddings"]
    dim = next((x.shape[0] for x in emb_list if x is not None), 512)
    arr = np.zeros((len(emb_list), dim), dtype=np.float32)
    for i, emb in enumerate(emb_list):
        if emb is not None:
            arr[i] = emb
    _atomic_torch_save(
        {
            "image_ids": payload["image_ids"],
            "identities": payload["identities"],
            "ok": payload["ok"],
            "embeddings": torch.from_numpy(arr),
            "complete": complete,
        },
        out,
    )


def compute_multinoise_embeddings(
    records: list[CelebARecord],
    paths: Paths,
    batch_size: int = 512,
    ctx_id: int = 0,
    overwrite: bool = False,
    save_every: int = 16,
) -> None:
    import cv2

    app = load_face_app(ctx_id=ctx_id)
    progress = progress_bar(total=len(records), desc="stage0 buffalo_l embeddings")
    for start in range(0, len(records), batch_size):
        shard_idx = start // batch_size
        chunk = records[start : start + batch_size]
        expected = [_shard_path(paths.embeddings_dir, level, shard_idx) for level in NOISE_LEVELS]
        if not overwrite and all(_is_complete_shard(p, len(chunk)) for p in expected):
            progress.update(len(chunk))
            continue

        by_level: dict[int, dict[str, object]] = {
            level: {"image_ids": [], "embeddings": [], "identities": [], "ok": []} for level in NOISE_LEVELS
        }
        if not overwrite:
            for level, out in zip(NOISE_LEVELS, expected):
                partial = _load_partial_shard(out)
                if partial is None or bool(partial.get("complete", False)):
                    continue
                n = min(len(partial.get("image_ids", [])), len(chunk))
                if [r.image_id for r in chunk[:n]] != list(partial.get("image_ids", []))[:n]:
                    continue
                embeddings = partial["embeddings"].float().numpy()
                by_level[level] = {
                    "image_ids": list(partial["image_ids"])[:n],
                    "identities": list(partial["identities"])[:n],
                    "ok": list(partial["ok"])[:n],
                    "embeddings": [
                        embeddings[i].astype(np.float32) if bool(partial["ok"][i]) else None for i in range(n)
                    ],
                }

        resume_idx = min(len(by_level[level]["image_ids"]) for level in NOISE_LEVELS)
        for level in NOISE_LEVELS:
            for key in ("image_ids", "embeddings", "identities", "ok"):
                by_level[level][key] = by_level[level][key][:resume_idx]
        if resume_idx:
            progress.update(resume_idx)

        for local_idx, record in enumerate(chunk[resume_idx:], start=resume_idx):
            image = cv2.imread(str(record.image_path))
            if image is None:
                for level in NOISE_LEVELS:
                    by_level[level]["image_ids"].append(record.image_id)
                    by_level[level]["embeddings"].append(None)
                    by_level[level]["identities"].append(record.identity)
                    by_level[level]["ok"].append(False)
            else:
                for level in NOISE_LEVELS:
                    noisy = add_gaussian_noise_bgr(image, level, record.image_id)
                    emb = embed_image_bgr(app, noisy)
                    by_level[level]["image_ids"].append(record.image_id)
                    by_level[level]["embeddings"].append(emb)
                    by_level[level]["identities"].append(record.identity)
                    by_level[level]["ok"].append(emb is not None)
            progress.update(1)

            should_save = ((local_idx + 1) % save_every == 0) or (local_idx + 1 == len(chunk))
            if should_save:
                complete = local_idx + 1 == len(chunk)
                for level, payload in by_level.items():
                    _payload_to_disk(payload, _shard_path(paths.embeddings_dir, level, shard_idx), complete=complete)

    progress.close()


def load_embedding_table(paths: Paths, level: int = 0) -> dict[str, object]:
    files = sorted((paths.embeddings_dir / f"noise_{level:02d}").glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"No embedding shards found for noise level {level} in {paths.embeddings_dir}")
    image_ids: list[str] = []
    identities: list[int | None] = []
    ok: list[bool] = []
    embeddings: list[torch.Tensor] = []
    for file in files:
        shard = torch.load(file, map_location="cpu")
        if not bool(shard.get("complete", True)):
            continue
        image_ids.extend(shard["image_ids"])
        identities.extend(shard["identities"])
        ok.extend(shard["ok"])
        embeddings.append(shard["embeddings"].float())
    embs = F.normalize(torch.cat(embeddings, dim=0), dim=1)
    return {"image_ids": image_ids, "identities": identities, "ok": ok, "embeddings": embs}


def export_level_embeddings(paths: Paths, level: int = 0, out_path: Path | None = None) -> Path:
    table = load_embedding_table(paths, level=level)
    ok = torch.tensor(table["ok"], dtype=torch.bool)
    out_path = out_path or (paths.embeddings_dir / f"buffalo_l_noise_{level:02d}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": [x for x, keep in zip(table["image_ids"], ok.tolist()) if keep],
            "identities": [x for x, keep in zip(table["identities"], ok.tolist()) if keep],
            "embeddings": table["embeddings"][ok],
        },
        out_path,
    )
    return out_path
