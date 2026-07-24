from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .celeba import load_records
from .config import Paths
from .embedding import embed_image_bgr, load_face_app


DEFAULT_PROMPT = (
    "A centered realistic half-body portrait photo of one person, upper body visible from head to waist, "
    "full face fully visible, entire head and shoulders visible, natural lighting, realistic skin texture"
)
DEFAULT_NEGATIVE = (
    "cropped face, partial face, face only, cut off head, cut off body, out of frame, extreme closeup, "
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
)
COMPARABLE_PROMPT = (
    "A centered realistic head-and-shoulders portrait photo of one person, "
    "full face fully visible, entire head visible, natural lighting, realistic skin texture"
)
COMPARABLE_NEGATIVE = (
    "cropped face, partial face, cut off head, out of frame, extreme closeup, "
    "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
)


def load_ip_adapter(paths: Paths, device: str = "cuda", lora_dir: Path | None = None):
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(paths.realvisxl_path),
        torch_dtype=torch.float16,
        scheduler=scheduler,
        add_watermarker=False,
    )
    ip_model = IPAdapterFaceIDXL(pipe, str(paths.ip_adapter_ckpt), device)
    if lora_dir is not None:
        full_adapter = Path(lora_dir) / "ip_adapter_full.pt"
        if full_adapter.exists():
            state = torch.load(full_adapter, map_location="cpu")
            ip_model.image_proj_model.load_state_dict(state["image_proj"])
            ip_layers = torch.nn.ModuleList(ip_model.pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state["ip_adapter"])
            ip_model.image_proj_model.to(device).eval()
            ip_layers.to(device).eval()
        else:
            from peft import PeftModel

            ip_model.pipe.unet = PeftModel.from_pretrained(ip_model.pipe.unet, str(lora_dir)).to(device)
            ip_model.pipe.unet.eval()
    return ip_model


def generate_from_embedding_file(
    paths: Paths,
    embedding_file: Path,
    split: str = "test",
    out_dir: Path | None = None,
    limit: int | None = None,
    width: int = 768,
    height: int = 1024,
    steps: int = 30,
    seed: int = 2023,
    device: str = "cuda",
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE,
    lora_dir: Path | None = None,
) -> Path:
    records = load_records(paths, require_identity=False)
    split_image_ids = [r.image_id for r in records if r.split == split]
    if limit is not None:
        split_image_ids = split_image_ids[:limit]
    split_ids = set(split_image_ids)
    payload = torch.load(embedding_file, map_location="cpu")
    image_ids: list[str] = payload["image_ids"]
    embeddings: torch.Tensor = payload["embeddings"].float()
    out_dir = out_dir or (paths.generated_dir / embedding_file.stem / split)
    out_dir.mkdir(parents=True, exist_ok=True)
    ip_model = load_ip_adapter(paths, device=device, lora_dir=lora_dir)
    count = 0
    for image_id, emb in zip(image_ids, embeddings):
        if image_id not in split_ids:
            continue
        out_path = out_dir / image_id
        if out_path.exists():
            count += 1
            if limit is not None and count >= limit:
                break
            continue
        faceid_embeds = emb.unsqueeze(0).to(device=device, dtype=torch.float16)
        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=7.5,
            seed=seed,
        )
        images[0].save(out_path)
        count += 1
        if limit is not None and count >= limit:
            break
    return out_dir


def evaluate_reconstructions(
    paths: Paths,
    generated_dir: Path,
    split: str = "test",
    limit: int | None = None,
    device: str = "cuda",
    out_json: Path | None = None,
) -> dict[str, float]:
    import cv2
    from transformers import CLIPImageProcessor, CLIPModel

    records = [r for r in load_records(paths, require_identity=False) if r.split == split]
    if limit is not None:
        records = records[:limit]
    app = load_face_app(ctx_id=0 if device.startswith("cuda") else -1)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    insight_scores = []
    clip_scores = []
    for record in records:
        pred_path = generated_dir / record.image_id
        if not pred_path.exists():
            continue
        real_bgr = cv2.imread(str(record.image_path))
        pred_bgr = cv2.imread(str(pred_path))
        if real_bgr is None or pred_bgr is None:
            continue
        real_emb = embed_image_bgr(app, real_bgr)
        pred_emb = embed_image_bgr(app, pred_bgr)
        if real_emb is not None and pred_emb is not None:
            a = torch.from_numpy(real_emb)
            b = torch.from_numpy(pred_emb)
            insight_scores.append(F.cosine_similarity(a, b, dim=0).item())

        real_img = Image.open(record.image_path).convert("RGB")
        pred_img = Image.open(pred_path).convert("RGB")
        inputs = clip_processor(images=[real_img, pred_img], return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
            feats = F.normalize(feats.float(), dim=1)
        clip_scores.append((feats[0] * feats[1]).sum().item())

    metrics = {
        "num_pairs": float(len(clip_scores)),
        "insightface_cosine_mean": float(sum(insight_scores) / max(1, len(insight_scores))),
        "clip_vitl14_cosine_mean": float(sum(clip_scores) / max(1, len(clip_scores))),
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics
