#!/usr/bin/env python3
"""
下载 RealVisXL V3.0、IP-Adapter-FaceID (SDXL) 和 CLIP ViT-L/14 的脚本。
所有文件下载到本脚本同级目录下的 models/ 文件夹。
已存在的模型自动跳过。
"""

import sys
import subprocess
from pathlib import Path

# ---------- 配置 ----------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 各模型目录 & 存在性哨兵文件
MODELS = {
    "RealVisXL V3.0": {
        "dir": MODELS_DIR / "realvisxl-v3.0",
        "sentinel": "model_index.json",
    },
    "IP-Adapter-FaceID": {
        "dir": MODELS_DIR / "ip-adapter-faceid",
        "sentinel": "ip-adapter-faceid_sdxl.bin",
    },
    "CLIP ViT-L/14": {
        "dir": MODELS_DIR / "clip-vit-large-patch14",
        "sentinel": "model.safetensors",   # 或 pytorch_model.bin
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def exists(model_key: str) -> bool:
    """检查模型是否已下载（通过哨兵文件判断）。"""
    m = MODELS[model_key]
    sentinel = m["dir"] / m["sentinel"]
    # 也支持 glob 模式（如 CLIP 的 safetensors 可能带有 hash 前缀）
    if "*" in m["sentinel"]:
        return any(m["dir"].glob(m["sentinel"]))
    return sentinel.exists()


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f">>> {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"[ERROR] {desc} 失败，返回码: {result.returncode}")
        sys.exit(1)


# ===================== RealVisXL V3.0 =====================
def download_realvisxl():
    key = "RealVisXL V3.0"
    d = MODELS[key]["dir"]

    print(f"\n{'='*60}")
    print(f">>> 1/3  下载 {key} (SG161222/RealVisXL_V3.0)")
    print(f"{'='*60}")

    if exists(key):
        print(f"[SKIP] {d} 已存在，跳过。")
        return

    ensure_dir(d)

    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "diffusers", "transformers", "accelerate", "-q"],
                "安装 diffusers / transformers / accelerate")
        from diffusers import DiffusionPipeline
    import torch

    print(f"[DOWNLOAD] 从 HuggingFace 下载 SG161222/RealVisXL_V3.0 → {d}")
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V3.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.save_pretrained(d)
    print("[OK] 已保存。")
    del pipe

    print(f"[DONE] {key} 下载完成。")


# ===================== IP-Adapter-FaceID =====================
def download_ip_adapter_faceid():
    key = "IP-Adapter-FaceID"
    d = MODELS[key]["dir"]

    print(f"\n{'='*60}")
    print(f">>> 2/3  下载 {key} (h94/IP-Adapter-FaceID)")
    print(f"{'='*60}")

    if exists(key):
        print(f"[SKIP] {d} 已存在，跳过。")
        return

    ensure_dir(d)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"],
                "安装 huggingface_hub")
        from huggingface_hub import snapshot_download

    print(f"[DOWNLOAD] 从 HuggingFace 下载 h94/IP-Adapter-FaceID (仅 SDXL 文件) → {d}")
    snapshot_download(
        repo_id="h94/IP-Adapter-FaceID",
        local_dir=str(d),
        allow_patterns=["*sdxl*", "*.json"],
        ignore_patterns=["*sd15*", "*plus*", "*_lora*"],
    )
    print("[OK] 已保存。")

    print(f"[DONE] {key} 下载完成。")


# ===================== CLIP ViT-L/14 =====================
def download_clip():
    key = "CLIP ViT-L/14"
    d = MODELS[key]["dir"]

    print(f"\n{'='*60}")
    print(f">>> 3/3  下载 {key} (openai/clip-vit-large-patch14)")
    print(f"{'='*60}")

    if exists(key):
        print(f"[SKIP] {d} 已存在，跳过。")
        return

    ensure_dir(d)

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "transformers", "-q"],
                "安装 transformers")
        from transformers import CLIPModel, CLIPProcessor

    print(f"[DOWNLOAD] 从 HuggingFace 下载 openai/clip-vit-large-patch14 → {d}")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model.save_pretrained(d)
    processor.save_pretrained(d)
    print("[OK] 已保存。")
    del model, processor

    print(f"[DONE] {key} 下载完成。")


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("   模型下载脚本")
    print(f"   目标目录: {MODELS_DIR}")
    print("=" * 60)

    # ---- 存在性检查 ----
    print("\n[CHECK] 检查已有模型...")
    for name in MODELS:
        status = "✓ 已存在" if exists(name) else "✗ 需下载"
        print(f"  {status}  {name}")

    ensure_dir(MODELS_DIR)

    download_realvisxl()
    download_ip_adapter_faceid()
    download_clip()

    print("\n" + "=" * 60)
    print("  全部完成！")
    for name, m in MODELS.items():
        print(f"  {name:<24} → {m['dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
