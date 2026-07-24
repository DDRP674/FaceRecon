#!/bin/bash
set -euo pipefail

set +u
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate face
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

configure_cuda_library_path() {
  local cuda_libs
  cuda_libs="$(
    python - <<'PY'
import os
import site

parts = []
roots = site.getsitepackages() + [site.getusersitepackages()]
for root in roots:
    for rel in [
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_nvrtc/lib",
        "nvidia/cufft/lib",
        "nvidia/curand/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "nvidia/nccl/lib",
        "nvidia/nvtx/lib",
    ]:
        path = os.path.join(root, rel)
        if os.path.isdir(path):
            parts.append(path)
print(":".join(parts))
PY
  )"
  if [[ -n "${cuda_libs}" ]]; then
    export LD_LIBRARY_PATH="${cuda_libs}:${LD_LIBRARY_PATH:-}"
  fi
}

ensure_cuda_compatible_torch() {
  if python - <<'PY'
import torch
cuda_build = torch.version.cuda or ""
major = int(cuda_build.split(".", 1)[0]) if cuda_build else 0
raise SystemExit(0 if major == 12 and torch.cuda.is_available() else 1)
PY
  then
    log_step "PyTorch CUDA build is already compatible"
  else
    log_step "Installing CUDA 12.1 compatible PyTorch into conda env: ${CONDA_DEFAULT_ENV}"
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      --force-reinstall \
      "torch==2.4.1+cu121" \
      "torchvision==0.19.1+cu121" \
      "torchaudio==2.4.1+cu121"
    python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is still unavailable after installing cu121 PyTorch")
PY
  fi
}

ensure_diffusers_stack() {
  if python - <<'PY'
import importlib.metadata as md

required = {
    "diffusers": "0.30.3",
    "transformers": "4.44.2",
    "accelerate": "0.34.2",
    "huggingface_hub": "0.24.6",
}

for pkg, want in required.items():
    try:
        have = md.version(pkg)
    except md.PackageNotFoundError:
        raise SystemExit(1)
    if have != want:
        print(f"{pkg} needs {want}, found {have}")
        raise SystemExit(1)

from diffusers import AutoencoderKL
print("diffusers stack OK", flush=True)
PY
  then
    log_step "Diffusers stack is already compatible"
  else
    log_step "Installing compatible diffusers stack"
    python -m pip install \
      "diffusers==0.30.3" \
      "transformers==4.44.2" \
      "accelerate==0.34.2" \
      "huggingface_hub==0.24.6"
    python - <<'PY'
from diffusers import AutoencoderKL
import importlib.metadata as md
print(
    "diffusers", md.version("diffusers"),
    "transformers", md.version("transformers"),
    "accelerate", md.version("accelerate"),
    "huggingface_hub", md.version("huggingface_hub"),
    "AutoencoderKL import OK",
    flush=True,
)
PY
  fi
}

ensure_lora_stack() {
  if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("peft") is not None else 1)
PY
  then
    log_step "LoRA stack is already available"
  else
    log_step "Installing LoRA training dependency: peft"
    python -m pip install "peft==0.12.0"
    python - <<'PY'
import peft
print("peft import OK", flush=True)
PY
  fi
}

ensure_onnxruntime_gpu() {
  if python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime providers:", providers, flush=True)
raise SystemExit(0 if "CUDAExecutionProvider" in providers else 1)
PY
  then
    log_step "ONNXRuntime CUDAExecutionProvider is available"
  else
    log_step "Installing CUDA 12 ONNXRuntime GPU"
    python -m pip uninstall -y onnxruntime || true
    python -m pip install "onnxruntime-gpu==1.20.1"
    python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime providers:", providers, flush=True)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is still unavailable after installing onnxruntime-gpu")
PY
  fi
}

verify_buffalo_l_cuda() {
  log_step "Verifying buffalo_l recognition is actually using CUDAExecutionProvider"
  python - <<'PY'
from facereco.embedding import load_recognition_model

model = load_recognition_model(ctx_id=0)
providers = model.session.get_providers()
print("buffalo_l recognition providers:", providers, flush=True)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(f"buffalo_l recognition is not using CUDAExecutionProvider: {providers}")
PY
}

ensure_ip_adapter_assets() {
  log_step "Ensuring IP-Adapter source and FaceID XL checkpoint are available"
  mkdir -p ../models
  if [[ ! -d ../models/IP-Adapter/ip_adapter ]]; then
    if [[ -e ../models/IP-Adapter ]]; then
      echo "../models/IP-Adapter exists but does not contain ip_adapter/. Move it aside or replace it with the IP-Adapter source repo." >&2
      return 1
    fi
    git clone --depth 1 https://github.com/tencent-ailab/IP-Adapter.git ../models/IP-Adapter
  fi
  export PYTHONPATH="$(pwd)/../models/IP-Adapter:${PYTHONPATH:-}"

  if [[ ! -f ../models/ip-adapter-faceid/ip-adapter-faceid_sdxl.bin ]]; then
    mkdir -p ../models/ip-adapter-faceid
    python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="h94/IP-Adapter-FaceID",
    filename="ip-adapter-faceid_sdxl.bin",
    local_dir="../models/ip-adapter-faceid",
    local_dir_use_symlinks=False,
)
print(path, flush=True)
PY
  fi

  python - <<'PY'
import importlib.util
if importlib.util.find_spec("ip_adapter") is None:
    raise SystemExit("ip_adapter package is still not importable after downloading source")
print("ip_adapter import OK", flush=True)
PY
}

has_identity_labels() {
  [[ -f ../data/celeba/identity_CelebA.txt ]] ||
    [[ -f ../data/celeba/identity_CelebA.csv ]] ||
    [[ -f ../data/celeba/identity_celeba.txt ]] ||
    [[ -f ../data/celeba/identity_celeba.csv ]] ||
    [[ -f ../data/celeba/list_identity_celeba.txt ]] ||
    [[ -f ../data/celeba/list_identity_celeba.csv ]]
}

has_reconstruction_deps() {
  python - <<'PY'
import importlib.util
mods = ["cv2", "diffusers", "ip_adapter", "transformers"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("missing reconstruction deps:", ", ".join(missing))
    raise SystemExit(1)
raise SystemExit(0)
PY
}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_optional() {
  log_step "$1"
  shift
  if "$@"; then
    log_step "Finished optional step"
  else
    log_step "Optional step failed or was interrupted; continuing with saved outputs"
  fi
}

ensure_cuda_compatible_torch
ensure_diffusers_stack
ensure_lora_stack
configure_cuda_library_path
ensure_onnxruntime_gpu
verify_buffalo_l_cuda
ensure_ip_adapter_assets

log_step "Writing official CelebA split manifest"
python run_pipeline.py split

log_step "Exporting original 0 percent buffalo_l embeddings"
python run_pipeline.py export-level --level 0

if has_identity_labels; then
  if has_reconstruction_deps; then
    if [[ ! -f ../work/embeddings/defended_train.pt || ! -f ../work/embeddings/defended_test.pt ]]; then
      echo "Missing defended embeddings. Run Stage 3 export before Stage 5." >&2
      exit 1
    fi
    if [[ ! -f ../work/embeddings/defended_val.pt ]]; then
      log_step "Exporting defended validation embeddings for Stage 5 validation"
      python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split val
    fi

    run_optional "Stage 5: fine-tuning full IP-Adapter, attacking, and evaluating reconstruction" \
  python run_pipeline.py stage5 \
    --embedding-file ../work/embeddings/defended_train.pt \
    --val-embedding-file ../work/embeddings/defended_val.pt \
    --eval-embedding-file ../work/embeddings/defended_test.pt \
    --lora-dir ../work/checkpoints/ip_adapter_defended_fulladapter_rawimg_bs1_lr5e7_ep20 \
    --generated-dir ../work/generated/stage5_test200_768x1024_fulladapter_rawimg_bs1_lr5e7_ep20 \
    --metrics-out ../work/metrics/stage5_reconstruction_fulladapter_rawimg_bs1_lr5e7_ep20.json \
      --limit 200 \
      --width 768 \
      --height 1024 \
      --prompt "A centered realistic head-and-shoulders portrait photo of one person, full face fully visible, entire head visible, natural lighting, realistic skin texture" \
      --negative-prompt "cropped face, partial face, cut off head, out of frame, extreme closeup, monochrome, lowres, bad anatomy, worst quality, low quality, blurry" \
      --epochs 20 \
      --steps-per-epoch 1000 \
      --val-batches 50 \
      --batch-size 1 \
      --lr 5e-7 \
      --max-grad-norm 1.0 \
      --loss-smooth-window 100 \
      --generate \
      --evaluate

    if [[ -d ../work/generated/stage2_test200_768x1024 && -d ../work/generated/stage4_test200_768x1024 ]] &&
      find ../work/generated/stage5_test200_768x1024_fulladapter_rawimg_bs1_lr5e7_ep20 -maxdepth 1 -name '*.jpg' -print -quit | grep -q .; then
      log_step "Creating GT/Stage2/Stage4/Stage5 comparison sheet"
      python run_pipeline.py compare \
    --stage2-dir ../work/generated/stage2_test200_768x1024 \
    --stage4-dir ../work/generated/stage4_test200_768x1024 \
    --stage5-dir ../work/generated/stage5_test200_768x1024_fulladapter_rawimg_bs1_lr5e7_ep20 \
    --out ../work/comparisons/gt_stage2_stage4_stage5_fulladapter_rawimg_bs1_lr5e7_ep20.jpg \
        --limit 8
    fi
  else
    log_step "Skipping Stage 5 because reconstruction dependencies are not importable"
  fi
else
  log_step "Skipping Stage 5 because CelebA identity labels are missing"
fi
