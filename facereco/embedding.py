from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from .celeba import CelebARecord, read_identities, read_landmarks
from .config import NOISE_LEVELS, Paths
from .noise import add_gaussian_noise_bgr
from .progress import progress_bar


def load_face_app(ctx_id: int = 0, det_size: tuple[int, int] = (640, 640)):
    import torch  # Preload CUDA/cuDNN libs from the PyTorch wheel for ONNXRuntime.
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    if ctx_id >= 0 and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNXRuntime CUDAExecutionProvider is unavailable; refusing to run buffalo_l on CPU.")

    providers = ["CUDAExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    if ctx_id >= 0:
        for taskname, model in app.models.items():
            session = getattr(model, "session", None)
            if session is not None and "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError(
                    f"buffalo_l {taskname} session is not using CUDAExecutionProvider: {session.get_providers()}"
                )
    return app


def load_recognition_model(ctx_id: int = 0):
    import torch  # Preload CUDA/cuDNN libs from the PyTorch wheel for ONNXRuntime.
    from insightface import model_zoo
    from insightface.utils.storage import ensure_available
    import onnxruntime as ort

    if ctx_id >= 0 and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNXRuntime CUDAExecutionProvider is unavailable; refusing to run buffalo_l on CPU.")

    model_dir = ensure_available("models", "buffalo_l", root="~/.insightface")
    onnx_file = Path(model_dir) / "w600k_r50.onnx"
    if not onnx_file.exists():
        raise RuntimeError(f"buffalo_l recognition ONNX model not found: {onnx_file}")
    providers = ["CUDAExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
    model = model_zoo.get_model(str(onnx_file), providers=providers)
    if model is None or getattr(model, "taskname", None) != "recognition":
        raise RuntimeError(f"Unexpected buffalo_l recognition model: {onnx_file}")
    model.prepare(ctx_id=ctx_id)
    session = getattr(model, "session", None)
    if ctx_id >= 0 and (session is None or "CUDAExecutionProvider" not in session.get_providers()):
        providers_used = None if session is None else session.get_providers()
        raise RuntimeError(f"buffalo_l recognition session is not using CUDAExecutionProvider: {providers_used}")
    return model


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))


def embed_image_bgr(app, image_bgr: np.ndarray) -> np.ndarray | None:
    face = largest_face(app.get(image_bgr))
    if face is None:
        return None
    return np.asarray(face.normed_embedding, dtype=np.float32)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _largest_detection(det_model, image_bgr: np.ndarray):
    bboxes, kpss = det_model.detect(image_bgr, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return None, None
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    idx = int(np.argmax(areas))
    kps = None if kpss is None else kpss[idx]
    return bboxes[idx], kps


def embed_multinoise_fast(app, image_bgr: np.ndarray, image_id: str) -> dict[int, np.ndarray | None]:
    from insightface.utils import face_align

    rec_model = app.models.get("recognition")
    if rec_model is None:
        raise RuntimeError("buffalo_l recognition model is not loaded")

    _bbox, kps = _largest_detection(app.det_model, image_bgr)
    if kps is None:
        return {level: None for level in NOISE_LEVELS}

    aligned = face_align.norm_crop(image_bgr, landmark=kps, image_size=rec_model.input_size[0])
    crops = [add_gaussian_noise_bgr(aligned, level, f"{image_id}:aligned") for level in NOISE_LEVELS]
    embeddings = _normalize_embeddings(rec_model.get_feat(crops))
    return {level: embeddings[i] for i, level in enumerate(NOISE_LEVELS)}


def embed_multinoise_landmarks(
    image_bgr: np.ndarray,
    image_id: str,
    landmarks: list[list[float]] | np.ndarray | None,
    image_size: int = 112,
) -> list[np.ndarray] | None:
    from insightface.utils import face_align

    if landmarks is None:
        return None
    kps = np.asarray(landmarks, dtype=np.float32)
    aligned = face_align.norm_crop(image_bgr, landmark=kps, image_size=image_size)
    return [add_gaussian_noise_bgr(aligned, level, f"{image_id}:landmarks") for level in NOISE_LEVELS]


def embed_multinoise_full(app, image_bgr: np.ndarray, image_id: str) -> dict[int, np.ndarray | None]:
    out = {}
    for level in NOISE_LEVELS:
        noisy = add_gaussian_noise_bgr(image_bgr, level, image_id)
        out[level] = embed_image_bgr(app, noisy)
    return out


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


def _is_complete_shard(out: Path, expected_len: int, embedding_mode: str | None = None) -> bool:
    payload = _load_partial_shard(out)
    if payload is None:
        return False
    image_ids = payload.get("image_ids", [])
    if len(image_ids) != expected_len:
        return False
    if embedding_mode is not None and payload.get("embedding_mode") != embedding_mode:
        return False
    return bool(payload.get("complete", True))


def _payload_to_disk(payload: dict[str, object], out: Path, complete: bool, embedding_mode: str) -> None:
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
            "embedding_mode": embedding_mode,
        },
        out,
    )


def _init_by_level() -> dict[int, dict[str, object]]:
    return {level: {"image_ids": [], "embeddings": [], "identities": [], "ok": []} for level in NOISE_LEVELS}


def _load_partial_by_level(
    paths: Paths,
    chunk: list[CelebARecord],
    shard_idx: int,
    embedding_mode: str,
    overwrite: bool,
) -> dict[int, dict[str, object]]:
    by_level = _init_by_level()
    if overwrite:
        return by_level
    for level in NOISE_LEVELS:
        out = _shard_path(paths.embeddings_dir, level, shard_idx)
        partial = _load_partial_shard(out)
        if partial is None or bool(partial.get("complete", False)):
            continue
        if partial.get("embedding_mode") != embedding_mode:
            continue
        n = min(len(partial.get("image_ids", [])), len(chunk))
        if [r.image_id for r in chunk[:n]] != list(partial.get("image_ids", []))[:n]:
            continue
        embeddings = partial["embeddings"].float().numpy()
        by_level[level] = {
            "image_ids": list(partial["image_ids"])[:n],
            "identities": list(partial["identities"])[:n],
            "ok": list(partial["ok"])[:n],
            "embeddings": [embeddings[i].astype(np.float32) if bool(partial["ok"][i]) else None for i in range(n)],
        }
    resume_idx = min(len(by_level[level]["image_ids"]) for level in NOISE_LEVELS)
    for level in NOISE_LEVELS:
        for key in ("image_ids", "embeddings", "identities", "ok"):
            by_level[level][key] = by_level[level][key][:resume_idx]
    return by_level


def _append_missing(by_level: dict[int, dict[str, object]], record: CelebARecord) -> None:
    for level in NOISE_LEVELS:
        by_level[level]["image_ids"].append(record.image_id)
        by_level[level]["embeddings"].append(None)
        by_level[level]["identities"].append(record.identity)
        by_level[level]["ok"].append(False)


def _flush_payloads(
    paths: Paths,
    by_level: dict[int, dict[str, object]],
    shard_idx: int,
    complete: bool,
    embedding_mode: str,
) -> None:
    for level, payload in by_level.items():
        _payload_to_disk(
            payload,
            _shard_path(paths.embeddings_dir, level, shard_idx),
            complete=complete,
            embedding_mode=embedding_mode,
        )


def _run_recognition_batches(rec_model, crops: list[np.ndarray], recognition_batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(crops), recognition_batch_size):
        outputs.append(_run_recognition_batch_with_retry(rec_model, crops[start : start + recognition_batch_size]))
    return _normalize_embeddings(np.concatenate(outputs, axis=0))


def _run_recognition_batch_with_retry(rec_model, crops: list[np.ndarray]) -> np.ndarray:
    import sys

    try:
        return rec_model.get_feat(crops)
    except Exception as exc:
        if len(crops) <= 1:
            raise
        mid = len(crops) // 2
        print(
            f"buffalo_l recognition batch of {len(crops)} failed ({type(exc).__name__}); "
            f"retrying as {mid}+{len(crops) - mid}",
            file=sys.stderr,
            flush=True,
        )
        left = _run_recognition_batch_with_retry(rec_model, crops[:mid])
        right = _run_recognition_batch_with_retry(rec_model, crops[mid:])
        return np.concatenate([left, right], axis=0)


def _prepare_landmark_crops(record: CelebARecord, landmarks: dict[str, list[list[float]]], image_size: int):
    import cv2

    image = cv2.imread(str(record.image_path))
    crop_list = None if image is None else embed_multinoise_landmarks(
        image,
        record.image_id,
        landmarks.get(record.image_id),
        image_size=image_size,
    )
    return crop_list


def _compute_landmark_embeddings(
    records: list[CelebARecord],
    paths: Paths,
    batch_size: int,
    ctx_id: int,
    det_size: int,
    overwrite: bool,
    save_every: int,
    recognition_batch_size: int,
    preprocess_workers: int,
    shard_index_offset: int,
) -> None:
    landmarks = read_landmarks(paths.data_dir)
    rec_model = load_recognition_model(ctx_id=ctx_id)
    face_size = int(rec_model.input_size[0])

    progress = progress_bar(total=len(records), desc="stage0 buffalo_l embeddings (landmarks)")
    for start in range(0, len(records), batch_size):
        shard_idx = shard_index_offset + start // batch_size
        chunk = records[start : start + batch_size]
        expected = [_shard_path(paths.embeddings_dir, level, shard_idx) for level in NOISE_LEVELS]
        if not overwrite and all(_is_complete_shard(p, len(chunk), "landmarks") for p in expected):
            progress.update(len(chunk))
            continue

        by_level = _load_partial_by_level(paths, chunk, shard_idx, "landmarks", overwrite)
        resume_idx = min(len(by_level[level]["image_ids"]) for level in NOISE_LEVELS)
        if resume_idx:
            progress.update(resume_idx)

        local_idx = resume_idx
        while local_idx < len(chunk):
            block = chunk[local_idx : min(len(chunk), local_idx + save_every)]
            crops: list[np.ndarray] = []
            crop_meta: list[tuple[int, int]] = []
            valid_offsets: set[int] = set()
            if preprocess_workers > 1:
                with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                    prepared = list(executor.map(lambda r: _prepare_landmark_crops(r, landmarks, face_size), block))
            else:
                prepared = [_prepare_landmark_crops(record, landmarks, face_size) for record in block]

            for offset, crop_list in enumerate(prepared):
                if crop_list is None:
                    continue
                valid_offsets.add(offset)
                for level_pos, crop in enumerate(crop_list):
                    crop_meta.append((offset, level_pos))
                    crops.append(crop)

            embeddings_by_record: dict[int, dict[int, np.ndarray]] = {i: {} for i in range(len(block))}
            if crops:
                embeddings = _run_recognition_batches(rec_model, crops, recognition_batch_size)
                for emb, (offset, level_pos) in zip(embeddings, crop_meta):
                    embeddings_by_record[offset][NOISE_LEVELS[level_pos]] = emb

            for offset, record in enumerate(block):
                if offset not in valid_offsets:
                    _append_missing(by_level, record)
                    continue
                for level in NOISE_LEVELS:
                    emb = embeddings_by_record[offset].get(level)
                    by_level[level]["image_ids"].append(record.image_id)
                    by_level[level]["embeddings"].append(emb)
                    by_level[level]["identities"].append(record.identity)
                    by_level[level]["ok"].append(emb is not None)

            local_idx += len(block)
            progress.update(len(block))
            _flush_payloads(
                paths,
                by_level,
                shard_idx,
                complete=local_idx == len(chunk),
                embedding_mode="landmarks",
            )
    progress.close()


def compute_multinoise_embeddings(
    records: list[CelebARecord],
    paths: Paths,
    batch_size: int = 512,
    ctx_id: int = 0,
    det_size: int = 320,
    overwrite: bool = False,
    save_every: int = 16,
    embedding_mode: str = "fast",
    recognition_batch_size: int = 4096,
    preprocess_workers: int = 1,
    shard_index_offset: int = 0,
) -> None:
    import cv2

    if embedding_mode not in {"landmarks", "fast", "full"}:
        raise ValueError(f"Unknown embedding_mode: {embedding_mode}")
    if embedding_mode == "landmarks":
        _compute_landmark_embeddings(
            records=records,
            paths=paths,
            batch_size=batch_size,
            ctx_id=ctx_id,
            det_size=det_size,
            overwrite=overwrite,
            save_every=save_every,
            recognition_batch_size=recognition_batch_size,
            preprocess_workers=preprocess_workers,
            shard_index_offset=shard_index_offset,
        )
        return

    app = load_face_app(ctx_id=ctx_id, det_size=(det_size, det_size))
    progress = progress_bar(total=len(records), desc=f"stage0 buffalo_l embeddings ({embedding_mode})")
    for start in range(0, len(records), batch_size):
        shard_idx = shard_index_offset + start // batch_size
        chunk = records[start : start + batch_size]
        expected = [_shard_path(paths.embeddings_dir, level, shard_idx) for level in NOISE_LEVELS]
        if not overwrite and all(_is_complete_shard(p, len(chunk), embedding_mode) for p in expected):
            progress.update(len(chunk))
            continue

        by_level = _load_partial_by_level(paths, chunk, shard_idx, embedding_mode, overwrite)

        resume_idx = min(len(by_level[level]["image_ids"]) for level in NOISE_LEVELS)
        for level in NOISE_LEVELS:
            for key in ("image_ids", "embeddings", "identities", "ok"):
                by_level[level][key] = by_level[level][key][:resume_idx]
        if resume_idx:
            progress.update(resume_idx)

        for local_idx, record in enumerate(chunk[resume_idx:], start=resume_idx):
            image = cv2.imread(str(record.image_path))
            if image is None:
                _append_missing(by_level, record)
            else:
                if embedding_mode == "fast":
                    embeddings_by_level = embed_multinoise_fast(app, image, record.image_id)
                else:
                    embeddings_by_level = embed_multinoise_full(app, image, record.image_id)
                for level in NOISE_LEVELS:
                    emb = embeddings_by_level[level]
                    by_level[level]["image_ids"].append(record.image_id)
                    by_level[level]["embeddings"].append(emb)
                    by_level[level]["identities"].append(record.identity)
                    by_level[level]["ok"].append(emb is not None)
            progress.update(1)

            should_save = ((local_idx + 1) % save_every == 0) or (local_idx + 1 == len(chunk))
            if should_save:
                _flush_payloads(
                    paths,
                    by_level,
                    shard_idx,
                    complete=local_idx + 1 == len(chunk),
                    embedding_mode=embedding_mode,
                )

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
    if any(identity is None for identity in identities):
        current_identities = read_identities(paths.data_dir)
        identities = [current_identities.get(image_id, identity) for image_id, identity in zip(image_ids, identities)]
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
