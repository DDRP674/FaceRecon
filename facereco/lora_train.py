from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .celeba import load_records
from .config import Paths


def _atomic_torch_save(payload: dict[str, object], out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


class FaceIdImageDataset(Dataset):
    def __init__(
        self,
        paths: Paths,
        embedding_file: Path,
        split: str = "train",
        image_size: int = 512,
        use_cached_latents: bool = True,
    ):
        records = load_records(paths, require_identity=True)
        record_by_id = {r.image_id: r for r in records if r.split == split}
        payload = torch.load(embedding_file, map_location="cpu")
        self.cache_file = paths.latents_dir / f"stage5_{embedding_file.stem}_{split}_vae{image_size}.pt"
        emb_by_id = {
            image_id: emb.float()
            for image_id, emb in zip(payload["image_ids"], payload["embeddings"])
            if image_id in record_by_id
        }
        self.items = []
        self.latents: torch.Tensor | None = None
        if use_cached_latents:
            self._load_cached_latents(paths, emb_by_id, image_size)
            if len(self.items) == 0:
                raise RuntimeError(
                    f"No cached SDXL VAE latents matched {split} embeddings. "
                    f"Expected files like {paths.latents_dir / f'sdxl_vae_{image_size}_shard_00000.pt'}."
                )
            return
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

    def _load_cached_latents(self, paths: Paths, emb_by_id: dict[str, torch.Tensor], image_size: int) -> None:
        if self.cache_file.exists():
            print(f"Loading Stage 5 latent dataset cache: {self.cache_file}", flush=True)
            payload = torch.load(self.cache_file, map_location="cpu")
            self.items = [(image_id, emb.float()) for image_id, emb in zip(payload["image_ids"], payload["embeddings"])]
            self.latents = payload["latents"].to(dtype=torch.float16).contiguous()
            return

        shards = sorted(paths.latents_dir.glob(f"sdxl_vae_{image_size}_shard_*.pt"))
        if not shards:
            raise RuntimeError(f"Missing cached SDXL VAE latents in {paths.latents_dir}; run Stage 0 VAE cache first.")
        print(f"Building Stage 5 latent dataset cache from {len(shards)} shards: {self.cache_file}", flush=True)
        latents = []
        for shard_idx, shard in enumerate(shards, start=1):
            if shard_idx == 1 or shard_idx % 1000 == 0 or shard_idx == len(shards):
                print(f"Stage 5 latent cache scan {shard_idx}/{len(shards)}", flush=True)
            payload = torch.load(shard, map_location="cpu")
            shard_ids = payload.get("image_ids", [])
            shard_latents = payload.get("latents")
            if shard_latents is None:
                continue
            for idx, image_id in enumerate(shard_ids):
                emb = emb_by_id.get(image_id)
                if emb is None:
                    continue
                self.items.append((image_id, emb))
                latents.append(shard_latents[idx].to(dtype=torch.float16))
        if latents:
            self.latents = torch.stack(latents, dim=0).contiguous()
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(
                {
                    "image_ids": [image_id for image_id, _emb in self.items],
                    "embeddings": torch.stack([emb for _image_id, emb in self.items], dim=0),
                    "latents": self.latents,
                },
                self.cache_file,
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self.latents is not None:
            image_id, emb = self.items[idx]
            return {"latents": self.latents[idx], "faceid_embeds": emb, "image_id": image_id}
        image_path, image_id, emb = self.items[idx]
        image = Image.open(image_path).convert("RGB")
        return {"pixel_values": self.tfm(image), "faceid_embeds": emb, "image_id": image_id}


class FP32MasterAdamW:
    def __init__(self, params: list[torch.nn.Parameter], lr: float, weight_decay: float = 1e-4, eps: float = 1e-6):
        self.params = [p for p in params if p.requires_grad]
        self.master_params = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        self.opt = torch.optim.AdamW(self.master_params, lr=lr, weight_decay=weight_decay, eps=eps)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None
        self.opt.zero_grad(set_to_none=True)

    def step(self, max_grad_norm: float) -> bool:
        for param, master in zip(self.params, self.master_params):
            if param.grad is None:
                master.grad = None
                continue
            grad = param.grad.detach().float()
            if not torch.isfinite(grad).all():
                self.opt.zero_grad(set_to_none=True)
                return False
            master.grad = grad
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.master_params, max_grad_norm)
        self.opt.step()
        with torch.no_grad():
            for param, master in zip(self.params, self.master_params):
                if not torch.isfinite(master).all():
                    for restore_param, restore_master in zip(self.params, self.master_params):
                        restore_master.copy_(restore_param.detach().float())
                    self.opt.zero_grad(set_to_none=True)
                    return False
                param.copy_(master.to(dtype=param.dtype))
        return True


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


def _adapter_layers(ip_model) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(ip_model.pipe.unet.attn_processors.values())


def _adapter_state_dict(ip_model) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "image_proj": ip_model.image_proj_model.state_dict(),
        "ip_adapter": _adapter_layers(ip_model).state_dict(),
    }


def _save_full_ip_adapter(ip_model, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_adapter_state_dict(ip_model), out_dir / "ip_adapter_full.pt")


def _encode_prompt(pipe, prompt: list[str], device: str):
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    text_inputs_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, return_tensors="pt")
    enc1 = pipe.text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
    enc2 = pipe.text_encoder_2(text_inputs_2.input_ids.to(device), output_hidden_states=True)
    prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
    return prompt_embeds, enc2[0]


def _get_prompt_condition(
    pipe,
    image_size: int,
    batch_size: int,
    device: str,
    prompt_text: str,
    prompt_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = prompt_cache.get(batch_size)
    if cached is not None:
        return cached
    prompt_embeds, pooled = _encode_prompt(pipe, [prompt_text] * batch_size, device)
    add_time_ids = torch.tensor([[image_size, image_size, 0, 0, image_size, image_size]], device=device, dtype=torch.float16)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    cached = (prompt_embeds, pooled, add_time_ids)
    prompt_cache[batch_size] = cached
    return cached


def _batch_loss(
    pipe,
    ip_model,
    scheduler,
    batch,
    image_size: int,
    device: str,
    prompt_text: str,
    prompt_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    faceid_embeds = batch["faceid_embeds"].to(device=device, dtype=torch.float16)
    with torch.no_grad():
        if "latents" in batch:
            latents = batch["latents"].to(device=device, dtype=torch.float16) * pipe.vae.config.scaling_factor
        else:
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float16)
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        prompt_embeds, pooled, add_time_ids = _get_prompt_condition(
            pipe,
            image_size,
            batch_size,
            device,
            prompt_text,
            prompt_cache,
        )

    if hasattr(ip_model, "get_image_embeds"):
        image_prompt_embeds = ip_model.image_proj_model(faceid_embeds)
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


def _read_step_losses(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values
    smoothed = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def _write_loss_curve(history: list[dict[str, float]], step_loss_path: Path, out_path: Path, smooth_window: int = 100) -> None:
    width, height = 900, 520
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.text((margin, 20), f"Stage 5 IP-Adapter loss, smoothed window={smooth_window}", fill="black")
    step_rows = _read_step_losses(step_loss_path)
    if not step_rows and not history:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return

    train_pairs = [
        (int(row["global_step"]), float(row["loss"]))
        for row in step_rows
        if row.get("loss") is not None and math.isfinite(float(row["loss"]))
    ]
    train_steps = [step for step, _value in train_pairs]
    train_values_raw = [value for _step, value in train_pairs]
    train_values = _smooth(train_values_raw, smooth_window)
    val_pairs = [
        (int(row["global_step"]), float(row["val_loss"]))
        for row in history
        if row.get("val_loss") is not None and math.isfinite(float(row["val_loss"]))
    ]
    val_steps = [step for step, _value in val_pairs]
    val_values = [value for _step, value in val_pairs]
    values = train_values + val_values
    if not values:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return
    ymin, ymax = min(values), max(values)
    if ymin == ymax:
        ymax = ymin + 1.0
    xmax = max(train_steps + val_steps + [1])

    def point(step: int, value: float) -> tuple[int, int]:
        x = margin + int((step - 1) * (width - 2 * margin) / max(1, xmax - 1))
        y = height - margin - int((value - ymin) * (height - 2 * margin) / (ymax - ymin))
        return x, y

    train_pts = [point(step, value) for step, value in zip(train_steps, train_values)]
    val_pts = [point(step, value) for step, value in zip(val_steps, val_values)]
    if len(train_pts) > 1:
        draw.line(train_pts, fill="royalblue", width=3)
    if len(val_pts) > 1:
        draw.line(val_pts, fill="crimson", width=3)
    for x, y in train_pts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="royalblue")
    for x, y in val_pts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="crimson")
    draw.text((width - 210, margin), "train smoothed", fill="royalblue")
    draw.text((width - 210, margin + 24), "validation", fill="crimson")
    draw.text((margin, height - margin + 18), "global step", fill="black")
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
    epochs: int = 20,
    steps_per_epoch: int = 1000,
    val_batches: int = 50,
    lr: float = 5e-7,
    max_grad_norm: float = 1.0,
    preload_latents_to_gpu: bool = True,
    use_cached_latents: bool = False,
    loss_smooth_window: int = 100,
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
    adapter_layers = _adapter_layers(ip_model)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    ip_model.image_proj_model.requires_grad_(True)
    adapter_layers.requires_grad_(True)
    ip_model.image_proj_model.train()
    adapter_layers.train()
    pipe.unet.train()

    dataset = FaceIdImageDataset(
        paths,
        embedding_file=embedding_file,
        split="train",
        image_size=image_size,
        use_cached_latents=use_cached_latents,
    )
    if len(dataset) == 0:
        raise RuntimeError("No Stage 5 training pairs found. Export defended train embeddings first.")
    if preload_latents_to_gpu and dataset.latents is not None:
        print(f"Preloading train latents to {device}: {tuple(dataset.latents.shape)}", flush=True)
        dataset.latents = dataset.latents.to(device=device, non_blocking=True)
    loader_workers = 0 if dataset.latents is not None else 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=dataset.latents is None,
        drop_last=True,
    )
    val_loader = None
    if val_embedding_file is not None:
        val_dataset = FaceIdImageDataset(
            paths,
            embedding_file=val_embedding_file,
            split="val",
            image_size=image_size,
            use_cached_latents=use_cached_latents,
        )
        if len(val_dataset) == 0:
            raise RuntimeError("No Stage 5 validation pairs found. Export defended val embeddings first.")
        if preload_latents_to_gpu and val_dataset.latents is not None:
            print(f"Preloading validation latents to {device}: {tuple(val_dataset.latents.shape)}", flush=True)
            val_dataset.latents = val_dataset.latents.to(device=device, non_blocking=True)
        val_workers = 0 if val_dataset.latents is not None else 2
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=val_dataset.latents is None,
        )
    trainable_params = list(ip_model.image_proj_model.parameters()) + list(adapter_layers.parameters())
    opt = FP32MasterAdamW(trainable_params, lr=lr, weight_decay=1e-4, eps=1e-6)

    prompt_text = "A centered realistic half-body portrait photo of one person"
    prompt_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    best_val = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    step_loss_path = out_dir / "lora_step_losses.jsonl"
    step_loss_path.write_text("")
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
            loss = _batch_loss(pipe, ip_model, scheduler, batch, image_size, device, prompt_text, prompt_cache)
            if not torch.isfinite(loss):
                global_step += 1
                step_row = {"epoch": epoch, "global_step": global_step, "loss": None, "skipped": True}
                with step_loss_path.open("a") as f:
                    f.write(json.dumps(step_row) + "\n")
                    f.flush()
                continue
            opt.zero_grad()
            loss.backward()
            stepped = opt.step(max_grad_norm)
            global_step += 1
            step_row = {"epoch": epoch, "global_step": global_step, "loss": loss.item(), "skipped": not stepped}
            with step_loss_path.open("a") as f:
                f.write(json.dumps(step_row) + "\n")
                f.flush()
            if stepped:
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
                    value = _batch_loss(pipe, ip_model, scheduler, batch, image_size, device, prompt_text, prompt_cache).item()
                    if math.isfinite(value):
                        vals.append(value)
            val_loss = None if not vals else sum(vals) / len(vals)
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                _save_full_ip_adapter(ip_model, out_dir)
        elif epoch == epochs:
            _save_full_ip_adapter(ip_model, out_dir)

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
        _write_loss_curve(history, step_loss_path, out_dir / "lora_loss_curve.png", smooth_window=loss_smooth_window)

    (out_dir / "lora_loss_history.json").write_text(json.dumps(history, indent=2))
    _write_loss_curve(history, step_loss_path, out_dir / "lora_loss_curve.png", smooth_window=loss_smooth_window)
    return out_dir
