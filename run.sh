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

log_step "Writing official CelebA split manifest"
python run_pipeline.py split

log_step "Stage 0A: caching SDXL VAE latents first"
python run_pipeline.py stage0 \
  --skip-embeddings \
  --cache-vae \
  --vae-batch-size 8 \
  --vae-image-size 512 \
  --device cuda

log_step "Stage 0B: caching ten-level buffalo_l embeddings"
python run_pipeline.py stage0 \
  --batch-size 64 \
  --ctx-id 0 \
  --save-every 4 \
  --device cuda

log_step "Exporting original 0 percent buffalo_l embeddings"
python run_pipeline.py export-level --level 0

if has_reconstruction_deps; then
  run_optional "Stage 2: attacking original system" \
    python run_pipeline.py stage2 \
    --embedding-file ../work/embeddings/buffalo_l_noise_00.pt \
    --generated-dir ../work/generated/stage2 \
    --generate \
    --evaluate
else
  log_step "Skipping Stage 2 because reconstruction dependencies are not importable"
fi

if has_identity_labels; then
  run_optional "Stage 1: original retrieval metrics" \
    python run_pipeline.py stage1 --query-split test --gallery-split train --level 0

  log_step "Stage 3: training defense model"
  if python run_pipeline.py stage3 --epochs 10 --batch-size 512 --lr 1e-3; then
    log_step "Exporting defended train/test embeddings"
    python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split train
    python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split test

    if has_reconstruction_deps; then
      run_optional "Stage 4: attacking defended embeddings directly" \
        python run_pipeline.py stage4 \
        --embedding-file ../work/embeddings/defended_test.pt \
        --generated-dir ../work/generated/stage4 \
        --generate \
        --evaluate

      run_optional "Stage 5: training IP-Adapter LoRA on defended embeddings" \
        python run_pipeline.py stage5 \
        --embedding-file ../work/embeddings/defended_train.pt \
        --steps 1000 \
        --batch-size 1 \
        --lr 1e-4
    else
      log_step "Skipping Stage 4/5 because reconstruction dependencies are not importable"
    fi
  else
    log_step "Stage 3 failed; skipping defended embedding export and Stage 4/5"
  fi
else
  log_step "Skipping Stage 1/3/4/5 because CelebA identity labels are missing"
fi
