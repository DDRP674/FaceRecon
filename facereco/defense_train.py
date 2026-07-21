from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .celeba import load_records
from .config import NOISE_LEVELS, Paths
from .defense_model import ArcMarginProduct, MultiNoiseDefense
from .embedding import load_embedding_table


class MultiNoiseEmbeddingDataset(Dataset):
    def __init__(self, paths: Paths, split: str = "train"):
        records = load_records(paths, require_identity=True)
        split_by_id = {r.image_id: r.split for r in records}
        tables = [load_embedding_table(paths, level=level) for level in NOISE_LEVELS]
        base_ids = tables[0]["image_ids"]
        labels_raw = tables[0]["identities"]
        ok = torch.ones(len(base_ids), dtype=torch.bool)
        for table in tables:
            if table["image_ids"] != base_ids:
                raise RuntimeError("Embedding shards are not aligned across noise levels.")
            ok &= torch.tensor(table["ok"], dtype=torch.bool)
        split_mask = torch.tensor([split_by_id.get(image_id, "") == split for image_id in base_ids], dtype=torch.bool)
        has_label = torch.tensor([x is not None for x in labels_raw], dtype=torch.bool)
        keep = ok & split_mask & has_label
        raw_kept = [int(labels_raw[i]) for i in keep.nonzero(as_tuple=False).flatten().tolist()]
        unique = {identity: idx for idx, identity in enumerate(sorted(set(raw_kept)))}
        self.label_map = unique
        self.image_ids = [base_ids[i] for i in keep.nonzero(as_tuple=False).flatten().tolist()]
        self.labels = torch.tensor([unique[x] for x in raw_kept], dtype=torch.long)
        self.embeddings = torch.stack([table["embeddings"][keep] for table in tables], dim=1).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


def train_defense(
    paths: Paths,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cuda",
    out_ckpt: Path | None = None,
) -> Path:
    dataset = MultiNoiseEmbeddingDataset(paths, split="train")
    if len(dataset) == 0:
        raise RuntimeError("No training embeddings found. Run stage0 and provide identity labels.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    embed_dim = dataset.embeddings.shape[-1]
    model = MultiNoiseDefense(embed_dim=embed_dim, num_levels=len(NOISE_LEVELS)).to(device)
    head = ArcMarginProduct(embed_dim, len(dataset.label_map)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for embeds, labels in loader:
            embeds = embeds.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            defended = model(embeds)
            logits = head(defended, labels)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * labels.numel()
            total += labels.numel()
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, total)})

    out_ckpt = out_ckpt or (paths.checkpoints_dir / "defense_arcface.pt")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "label_map": dataset.label_map,
            "embed_dim": embed_dim,
            "noise_levels": NOISE_LEVELS,
            "history": history,
        },
        out_ckpt,
    )
    (out_ckpt.with_suffix(".json")).write_text(json.dumps(history, indent=2))
    return out_ckpt


def export_defended_embeddings(paths: Paths, ckpt_path: Path, split: str | None = None, device: str = "cuda") -> Path:
    dataset = MultiNoiseEmbeddingDataset(paths, split=split or "train")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MultiNoiseDefense(embed_dim=int(ckpt["embed_dim"]), num_levels=len(ckpt["noise_levels"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=2)
    outs = []
    with torch.no_grad():
        for embeds, _labels in loader:
            outs.append(model(embeds.to(device)).cpu())
    out = paths.embeddings_dir / f"defended_{split or 'train'}.pt"
    torch.save({"image_ids": dataset.image_ids, "labels": dataset.labels, "embeddings": torch.cat(outs, dim=0)}, out)
    return out
