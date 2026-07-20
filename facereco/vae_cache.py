from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .celeba import CelebARecord
from .config import Paths
from .progress import progress_bar


def _atomic_torch_save(payload: dict[str, object], out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def _is_complete(out: Path) -> bool:
    if not out.exists():
        return False
    try:
        payload = torch.load(out, map_location="cpu")
    except Exception:
        return False
    return bool(payload.get("complete", True))


def _complete_marker(paths: Paths, image_size: int, batch_size: int, num_records: int) -> Path:
    return paths.latents_dir / f"sdxl_vae_{image_size}_bs{batch_size}_n{num_records}.complete"


def _expected_num_shards(num_records: int, batch_size: int) -> int:
    return (num_records + batch_size - 1) // batch_size


def _has_complete_cache(paths: Paths, image_size: int, batch_size: int, num_records: int) -> bool:
    marker = _complete_marker(paths, image_size, batch_size, num_records)
    if marker.exists():
        return True
    expected = _expected_num_shards(num_records, batch_size)
    files = list(paths.latents_dir.glob(f"sdxl_vae_{image_size}_shard_*.pt"))
    if len(files) >= expected:
        marker.write_text(f"complete\nimage_size={image_size}\nbatch_size={batch_size}\nnum_records={num_records}\n")
        return True
    return False


def cache_sdxl_vae_latents(
    records: list[CelebARecord],
    paths: Paths,
    image_size: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    overwrite: bool = False,
) -> None:
    paths.latents_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and _has_complete_cache(paths, image_size, batch_size, len(records)):
        print(f"SDXL VAE latents already cached for {len(records)} records; skipping.", flush=True)
        return

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(paths.realvisxl_path / "vae", torch_dtype=dtype).to(device)
    vae.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    progress = progress_bar(total=len(records), desc=f"stage0 SDXL VAE latents {image_size}")
    for start in range(0, len(records), batch_size):
        shard_idx = start // batch_size
        out = paths.latents_dir / f"sdxl_vae_{image_size}_shard_{shard_idx:05d}.pt"
        chunk = records[start : start + batch_size]
        if _is_complete(out) and not overwrite:
            progress.update(len(chunk))
            continue
        images = []
        kept = []
        for record in chunk:
            try:
                image = Image.open(record.image_path).convert("RGB")
            except OSError:
                continue
            images.append(tfm(image))
            kept.append(record)
        if not images:
            progress.update(len(chunk))
            continue
        pixel_values = torch.stack(images).to(device=device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.mean.cpu()
        _atomic_torch_save(
            {
                "image_ids": [r.image_id for r in kept],
                "identities": [r.identity for r in kept],
                "latents": latents,
                "complete": True,
            },
            out,
        )
        progress.update(len(chunk))
    progress.close()
    if _has_complete_cache(paths, image_size, batch_size, len(records)):
        _complete_marker(paths, image_size, batch_size, len(records)).write_text(
            f"complete\nimage_size={image_size}\nbatch_size={batch_size}\nnum_records={len(records)}\n"
        )
