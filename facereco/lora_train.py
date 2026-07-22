from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .celeba import load_records
from .config import Paths


class FaceIdImageDataset(Dataset):
    def __init__(self, paths: Paths, embedding_file: Path, split: str = "train", image_size: int = 512):
        records = load_records(paths, require_identity=True)
        record_by_id = {r.image_id: r for r in records if r.split == split}
        payload = torch.load(embedding_file, map_location="cpu")
        self.items = []
        for image_id, emb in zip(payload["image_ids"], payload["embeddings"]):
            record = record_by_id.get(image_id)
            if record is not None:
                self.items.append((record.image_path, image_id, emb.float()))
        self.tfm = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        image_path, image_id, emb = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        return {"pixel_values": self.tfm(image), "faceid_embeds": emb, "image_id": image_id}


def _apply_unet_lora(unet, rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    targets = set()
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "Linear" and name.split(".")[-1] in {
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_q_lora",
            "to_k_lora",
            "to_v_lora",
            "to_out_lora",
        }:
            targets.add(name.split(".")[-1])
    if not targets:
        targets = {"to_q", "to_k", "to_v", "to_out.0"}
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=sorted(targets), lora_dropout=0.05, bias="none")
    return get_peft_model(unet, cfg)


def _encode_prompt(pipe, prompt: list[str], device: str):
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    text_inputs_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, return_tensors="pt")
    enc1 = pipe.text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    enc2 = pipe.text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
    prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
    return prompt_embeds, enc2[0]


def _batch_loss(pipe, ip_model, scheduler, batch, image_size: int, device: str, prompt_text: str) -> torch.Tensor:
    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float16)
    faceid_embeds = batch["faceid_embeds"].to(device=device, dtype=torch.float16)
    batch_size = pixel_values.shape[0]
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        prompt_embeds, pooled = _encode_prompt(pipe, [prompt_text] * batch_size, device)
        add_time_ids = torch.tensor([[image_size, image_size, 0, 0, image_size, image_size]], device=device, dtype=torch.float16)
        add_time_ids = add_time_ids.repeat(batch_size, 1)

    if hasattr(ip_model, "get_image_embeds"):
        image_prompt_embeds, _ = ip_model.get_image_embeds(faceid_embeds)
        encoder_hidden_states = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
    else:
        encoder_hidden_states = prompt_embeds

    model_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
    ).sample
    return F.mse_loss(model_pred.float(), noise.float())


def _write_loss_curve(history: list[dict[str, float]], out_path: Path) -> None:
    width, height = 900, 520
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((margin, 20), "Stage 5 LoRA loss", fill="black")
    if not history:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return

    values = [float(row["train_loss"]) for row in history]
    values += [float(row["val_loss"]) for row in history if row.get("val_loss") is not None]
    ymin, ymax = min(values), max(values)
    if ymin == ymax:
        ymax = ymin + 1.0

    def point(epoch: int, value: float) -> tuple[int, int]:
        x = margin + int((epoch - 1) * (width - 2 * margin) / max(1, len(history) - 1))
        y = height - margin - int((value - ymin) * (height - 2 * margin) / (ymax - ymin))
        return x, y

    train_pts = [point(int(row["epoch"]), float(row["train_loss"])) for row in history]
    val_pts = [
        point(int(row["epoch"]), float(row["val_loss"]))
        for row in history
        if row.get("val_loss") is not None
    ]
    if len(train_pts) > 1:
        draw.line(train_pts, fill="royalblue", width=3)
    if len(val_pts) > 1:
        draw.line(val_pts, fill="crimson", width=3)
    for x, y in train_pts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="royalblue")
    for x, y in val_pts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="crimson")
    draw.text((width - 210, margin), "train", fill="royalblue")
    draw.text((width - 210, margin + 24), "validation", fill="crimson")
    draw.text((margin, height - margin + 18), "epoch", fill="black")
    draw.text((8, margin), f"{ymax:.4f}", fill="black")
    draw.text((8, height - margin - 10), f"{ymin:.4f}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def train_ip_adapter_lora(
    paths: Paths,
    embedding_file: Path,
    val_embedding_file: Path | None = None,
    out_dir: Path | None = None,
    image_size: int = 512,
    batch_size: int = 1,
    epochs: int = 12,
    steps_per_epoch: int = 1000,
    val_batches: int = 50,
    lr: float = 1e-4,
    rank: int = 8,
    alpha: int = 8,
    device: str = "cuda",
) -> Path:
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

    out_dir = out_dir or (paths.checkpoints_dir / "ip_adapter_defended_lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    scheduler = DDIMScheduler.from_pretrained(paths.realvisxl_path / "scheduler")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(paths.realvisxl_path),
        torch_dtype=torch.float16,
        scheduler=scheduler,
        add_watermarker=False,
    ).to(device)
    ip_model = IPAdapterFaceIDXL(pipe, str(paths.ip_adapter_ckpt), device)
    pipe.unet = _apply_unet_lora(pipe.unet, rank=rank, alpha=alpha)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.train()

    dataset = FaceIdImageDataset(paths, embedding_file=embedding_file, split="train", image_size=image_size)
    if len(dataset) == 0:
        raise RuntimeError("No Stage 5 training pairs found. Export defended train embeddings first.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = None
    if val_embedding_file is not None:
        val_dataset = FaceIdImageDataset(paths, embedding_file=val_embedding_file, split="val", image_size=image_size)
        if len(val_dataset) == 0:
            raise RuntimeError("No Stage 5 validation pairs found. Export defended val embeddings first.")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=lr)

    prompt_text = "A centered realistic half-body portrait photo of one person"
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        pipe.unet.train()
        train_total = 0.0
        train_count = 0
        train_iter = iter(loader)
        for _ in range(steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            loss = _batch_loss(pipe, ip_model, scheduler, batch, image_size, device, prompt_text)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            train_total += loss.item()
            train_count += 1

        val_loss = None
        if val_loader is not None:
            pipe.unet.eval()
            vals = []
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    if idx >= val_batches:
                        break
                    vals.append(_batch_loss(pipe, ip_model, scheduler, batch, image_size, device, prompt_text).item())
            val_loss = sum(vals) / max(1, len(vals))
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                pipe.unet.save_pretrained(out_dir)
        elif epoch == epochs:
            pipe.unet.save_pretrained(out_dir)

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_total / max(1, train_count),
            "val_loss": val_loss,
            "best_epoch": best_epoch,
            "best_val_loss": None if best_val == float("inf") else best_val,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (out_dir / "lora_loss_history.json").write_text(json.dumps(history, indent=2))
        _write_loss_curve(history, out_dir / "lora_loss_curve.png")

    (out_dir / "lora_loss_history.json").write_text(json.dumps(history, indent=2))
    _write_loss_curve(history, out_dir / "lora_loss_curve.png")
    return out_dir
