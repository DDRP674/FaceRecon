from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .celeba import load_records
from .config import Paths
from .embedding import load_embedding_table


def _chunked_topk(q: torch.Tensor, g: torch.Tensor, k: int, chunk_size: int = 512) -> torch.Tensor:
    chunks = []
    kk = min(k, g.shape[0])
    for start in range(0, q.shape[0], chunk_size):
        sims = q[start : start + chunk_size] @ g.T
        chunks.append(sims.topk(k=kk, dim=1).indices.cpu())
    return torch.cat(chunks, dim=0)


def evaluate_retrieval(
    paths: Paths,
    query_split: str = "test",
    gallery_split: str = "train",
    level: int = 0,
    topk: tuple[int, ...] = (1, 5),
    out_json: Path | None = None,
) -> dict[str, float]:
    records = load_records(paths, require_identity=True)
    split_by_id = {r.image_id: r.split for r in records}
    table = load_embedding_table(paths, level=level)
    embs: torch.Tensor = table["embeddings"]
    image_ids: list[str] = table["image_ids"]
    identities = torch.tensor([int(x) if x is not None else -1 for x in table["identities"]], dtype=torch.long)
    ok = torch.tensor(table["ok"], dtype=torch.bool)

    splits = [split_by_id.get(image_id, "") for image_id in image_ids]
    query_mask = torch.tensor([s == query_split for s in splits], dtype=torch.bool) & ok & (identities >= 0)
    gallery_mask = torch.tensor([s == gallery_split for s in splits], dtype=torch.bool) & ok & (identities >= 0)
    if gallery_split == query_split:
        same_split_self = True
    else:
        same_split_self = False
    q_idx = query_mask.nonzero(as_tuple=False).flatten()
    g_idx = gallery_mask.nonzero(as_tuple=False).flatten()
    if len(q_idx) == 0 or len(g_idx) == 0:
        raise RuntimeError("No valid query/gallery embeddings. Run stage0 and check identity labels.")

    q = F.normalize(embs[q_idx], dim=1)
    g = F.normalize(embs[g_idx], dim=1)
    max_k = max(topk)
    if same_split_self:
        chunks = []
        kk = min(max_k, g.shape[0])
        for start in range(0, q.shape[0], 512):
            end = min(start + 512, q.shape[0])
            sims = q[start:end] @ g.T
            for row, original_idx in enumerate(q_idx[start:end]):
                same = (g_idx == original_idx).nonzero(as_tuple=False).flatten()
                if len(same):
                    sims[row, same[0]] = -10.0
            chunks.append(sims.topk(k=kk, dim=1).indices.cpu())
        nn = torch.cat(chunks, dim=0)
    else:
        nn = _chunked_topk(q, g, max_k)
    q_labels = identities[q_idx]
    g_labels = identities[g_idx]
    metrics: dict[str, float] = {
        "num_queries": float(len(q_idx)),
        "num_gallery": float(len(g_idx)),
    }
    for k in topk:
        kk = min(k, nn.shape[1])
        hit = (g_labels[nn[:, :kk]] == q_labels[:, None]).any(dim=1).float().mean().item()
        metrics[f"top{k}"] = hit

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics
