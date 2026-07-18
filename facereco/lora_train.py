from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
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


def train_ip_adapter_lora(
    paths: Paths,
    embedding_file: Path,
    out_dir: Path | None = None,
    image_size: int = 512,
    batch_size: int = 1,
    steps: int = 1000,
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
    opt = torch.optim.AdamW([p for p in pipe.unet.parameters() if p.requires_grad], lr=lr)

    prompt = ["A closeup portrait photo of a person"] * batch_size
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    text_inputs_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, return_tensors="pt")
    global_step = 0
    while global_step < steps:
        for batch in loader:
            if global_step >= steps:
                break
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float16)
            faceid_embeds = batch["faceid_embeds"].to(device=device, dtype=torch.float16)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                enc1 = pipe.text_encoder(text_inputs.input_ids[: latents.shape[0]].to(device), output_hidden_states=True)
                enc2 = pipe.text_encoder_2(text_inputs_2.input_ids[: latents.shape[0]].to(device), output_hidden_states=True)
                prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1)
                pooled = enc2[0]
                add_time_ids = torch.tensor([[image_size, image_size, 0, 0, image_size, image_size]], device=device, dtype=torch.float16)
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

            if hasattr(ip_model, "get_image_embeds"):
                image_prompt_embeds, _ = ip_model.get_image_embeds(faceid_embeds)
                encoder_hidden_states = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            else:
                encoder_hidden_states = prompt_embeds

            added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}
            model_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            loss = F.mse_loss(model_pred.float(), noise.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1

    pipe.unet.save_pretrained(out_dir)
    return out_dir

