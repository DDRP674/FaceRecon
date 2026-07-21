from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .celeba import load_records
from .config import Paths


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_reconstruction_comparison(
    paths: Paths,
    stage2_dir: Path,
    stage4_dir: Path,
    stage5_dir: Path,
    out_path: Path,
    split: str = "test",
    limit: int = 8,
    thumb_size: tuple[int, int] = (192, 256),
) -> Path:
    records = [r for r in load_records(paths, require_identity=False) if r.split == split]
    rows = []
    for record in records:
        s2 = stage2_dir / record.image_id
        s4 = stage4_dir / record.image_id
        s5 = stage5_dir / record.image_id
        if record.image_path.exists() and s2.exists() and s4.exists() and s5.exists():
            rows.append((record.image_id, record.image_path, s2, s4, s5))
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError("No comparable GT/Stage2/Stage4/Stage5 image rows found.")

    labels = ["GT", "Stage2", "Stage4", "Stage5"]
    label_h = 28
    id_w = 92
    width = id_w + thumb_size[0] * len(labels)
    height = label_h * 2 + thumb_size[1] * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        draw.text((id_w + col * thumb_size[0] + 8, 8), label, fill="black")
    for row_idx, (image_id, gt, s2, s4, s5) in enumerate(rows):
        y = label_h + row_idx * thumb_size[1]
        draw.text((8, y + 8), image_id, fill="black")
        for col, path in enumerate([gt, s2, s4, s5]):
            sheet.paste(_thumb(path, thumb_size), (id_w + col * thumb_size[0], y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
