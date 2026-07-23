from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


def _load_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def _select_pairs(before_path: Path, after_path: Path, limit: int) -> tuple[list[str], list[int | None], torch.Tensor, torch.Tensor]:
    before = _load_payload(before_path)
    after = _load_payload(after_path)

    before_ids = list(before["image_ids"])
    before_embs = before["embeddings"].float()
    before_identities = before.get("identities")
    before_by_id = {image_id: idx for idx, image_id in enumerate(before_ids)}

    ids: list[str] = []
    identities: list[int | None] = []
    before_rows = []
    after_rows = []
    for image_id, after_emb in zip(after["image_ids"], after["embeddings"]):
        idx = before_by_id.get(image_id)
        if idx is None:
            continue
        ids.append(image_id)
        if before_identities is None:
            identities.append(None)
        else:
            identities.append(None if before_identities[idx] is None else int(before_identities[idx]))
        before_rows.append(before_embs[idx])
        after_rows.append(after_emb.float())
        if len(ids) >= limit:
            break

    if not ids:
        raise RuntimeError("No shared image_ids between before and after embedding files.")
    return ids, identities, torch.stack(before_rows, dim=0), torch.stack(after_rows, dim=0)


def _pairwise_squared_distances(x: torch.Tensor) -> torch.Tensor:
    diff = x[:, None, :] - x[None, :, :]
    return diff.square().sum(dim=-1)


def _joint_probabilities(x: torch.Tensor, perplexity: float) -> torch.Tensor:
    n = x.shape[0]
    distances = _pairwise_squared_distances(x)
    target_entropy = math.log(perplexity)
    p_cond = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        beta = torch.tensor(1.0)
        beta_min = None
        beta_max = None
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        row = distances[i, mask]
        probs = torch.empty_like(row)
        for _ in range(50):
            probs = torch.exp(-row * beta)
            sum_probs = probs.sum().clamp_min(1e-12)
            entropy = torch.log(sum_probs) + beta * (row * probs).sum() / sum_probs
            diff = entropy.item() - target_entropy
            if abs(diff) < 1e-5:
                break
            if diff > 0:
                beta_min = beta.clone()
                beta = beta * 2 if beta_max is None else (beta + beta_max) / 2
            else:
                beta_max = beta.clone()
                beta = beta / 2 if beta_min is None else (beta + beta_min) / 2
        p_cond[i, mask] = probs / probs.sum().clamp_min(1e-12)
    p = (p_cond + p_cond.t()) / (2 * n)
    return p.clamp_min(1e-12)


def _run_torch_tsne(before: torch.Tensor, after: torch.Tensor, perplexity: float, seed: int, iterations: int, lr: float, device: str):
    x = torch.cat([before, after], dim=0)
    x = torch.nn.functional.normalize(x.float(), dim=1).to(device)
    perplexity = min(perplexity, max(5.0, (len(x) - 1) / 3))
    p = _joint_probabilities(x, perplexity) * 4
    torch.manual_seed(seed)
    y = (torch.randn((x.shape[0], 2), dtype=torch.float32, device=device) * 1e-4).requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    for step in range(iterations):
        opt.zero_grad(set_to_none=True)
        distances = _pairwise_squared_distances(y)
        q_num = 1.0 / (1.0 + distances)
        q_num.fill_diagonal_(0.0)
        q = (q_num / q_num.sum().clamp_min(1e-12)).clamp_min(1e-12)
        loss = (p * (p.log() - q.log())).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            y -= y.mean(dim=0, keepdim=True)
        if step == 250:
            p = p / 4
        if (step + 1) % 50 == 0 or step == 0 or step + 1 == iterations:
            print(f"torch t-SNE step {step + 1}/{iterations} loss={loss.item():.6f}", flush=True)
    coords = y.detach().cpu().tolist()
    return coords[: len(before)], coords[len(before) :]


def _run_tsne(
    before: torch.Tensor,
    after: torch.Tensor,
    perplexity: float,
    seed: int,
    backend: str,
    iterations: int,
    lr: float,
    device: str,
):
    if backend in {"auto", "sklearn"}:
        try:
            return _run_sklearn_tsne(before, after, perplexity, seed)
        except Exception:
            if backend == "sklearn":
                raise
    return _run_torch_tsne(before, after, perplexity, seed, iterations, lr, device)


def _run_sklearn_tsne(before: torch.Tensor, after: torch.Tensor, perplexity: float, seed: int):
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for t-SNE. Run this in the face environment, "
            "or install a NumPy-compatible scikit-learn there."
        ) from exc

    x = torch.cat([before, after], dim=0)
    x = torch.nn.functional.normalize(x, dim=1).numpy()
    perplexity = min(perplexity, max(5.0, (len(x) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(x)
    return coords[: len(before)].tolist(), coords[len(before) :].tolist()


def _palette(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    saturation = 0.62
    value = 0.88
    h = hue * 6
    c = value * saturation
    x = c * (1 - abs(h % 2 - 1))
    m = value - c
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def _scale_points(points, box: tuple[int, int, int, int], all_points) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    xs = [float(p[0]) for p in all_points]
    ys = [float(p[1]) for p in all_points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = 24
    width = right - left - 2 * pad
    height = bottom - top - 2 * pad
    out = []
    for x, y in points:
        px = left + pad + int((float(x) - xmin) * width / (xmax - xmin))
        py = bottom - pad - int((float(y) - ymin) * height / (ymax - ymin))
        out.append((px, py))
    return out


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    title: str,
    points,
    colors: list[tuple[int, int, int]],
    box: tuple[int, int, int, int],
    all_points,
    marker: str,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(40, 40, 40), width=2)
    draw.text((left, top - 30), title, fill=(20, 20, 20))
    scaled = _scale_points(points, box, all_points)
    radius = 5
    for (x, y), color in zip(scaled, colors):
        if marker == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))


def _draw_overlay(draw: ImageDraw.ImageDraw, before_points, after_points, colors, box, all_points) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(40, 40, 40), width=2)
    draw.text((left, top - 30), "Before -> After", fill=(20, 20, 20))
    before_scaled = _scale_points(before_points, box, all_points)
    after_scaled = _scale_points(after_points, box, all_points)
    for start, end, color in zip(before_scaled, after_scaled, colors):
        pale = tuple((c + 255) // 2 for c in color)
        draw.line((*start, *end), fill=pale, width=1)
    radius = 4
    for (x, y), color in zip(before_scaled, colors):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
    for (x, y), color in zip(after_scaled, colors):
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))


def _draw_figure(before_points, after_points, identities: list[int | None], out: Path, title: str) -> None:
    unique = {identity: idx for idx, identity in enumerate(sorted({x for x in identities if x is not None}))}
    colors = [_palette(unique[x]) if x is not None else (80, 80, 80) for x in identities]
    all_points = list(before_points) + list(after_points)

    width, height = 1800, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), title, fill=(0, 0, 0))
    draw.text((40, 52), "Circle = before defense, square = after defense. Colors use CelebA identity labels.", fill=(70, 70, 70))
    _draw_panel(draw, "Before defense", before_points, colors, (40, 110, 580, 660), all_points, "circle")
    _draw_panel(draw, "After defense", after_points, colors, (630, 110, 1170, 660), all_points, "square")
    _draw_overlay(draw, before_points, after_points, colors, (1220, 110, 1760, 660), all_points)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw t-SNE visualization for embeddings before and after defense.")
    parser.add_argument("--before", type=Path, default=Path("../work/embeddings/buffalo_l_noise_00.pt"))
    parser.add_argument("--after", type=Path, default=Path("../work/embeddings/defended_test.pt"))
    parser.add_argument("--out", type=Path, default=Path("../work/visualizations/defense_tsne_test200.png"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--backend", choices=["auto", "sklearn", "torch"], default="auto")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids, identities, before, after = _select_pairs(args.before, args.after, args.limit)
    before_points, after_points = _run_tsne(
        before,
        after,
        args.perplexity,
        args.seed,
        args.backend,
        args.iterations,
        args.lr,
        args.device,
    )
    title = f"t-SNE of {len(ids)} CelebA test embeddings before and after defense"
    _draw_figure(before_points, after_points, identities, args.out, title)
    print(args.out)


if __name__ == "__main__":
    main()
