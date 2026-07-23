# CelebA Face Recognition Defense and Reconstruction Pipeline

This repository contains a staged pipeline for studying identity recognition, reconstruction attacks, and embedding-space defenses on CelebA. The system first caches InsightFace buffalo_l embeddings for clean and noisy CelebA images, evaluates retrieval accuracy, reconstructs faces from embeddings with an SDXL FaceID IP-Adapter, trains a defended embedding model, and then evaluates two post-defense attack settings.

All implementation code lives in `code/`. Large data, cached embeddings, checkpoints, generated images, and metrics are stored outside this folder under `../data`, `../models`, and `../work`.

## Pipeline Overview

The pipeline is organized into six stages.

Stage 0 builds reusable caches. It applies ten noise levels to every CelebA image, extracts buffalo_l embeddings, and can optionally cache SDXL VAE latents.

Stage 1 evaluates the original buffalo_l retrieval system with top-1 and top-5 identity retrieval metrics.

Stage 2 attacks the original system by reconstructing test faces from original buffalo_l embeddings with SDXL + IP-Adapter FaceID, then evaluates InsightFace cosine similarity and CLIP ViT-L/14 cosine similarity.

Stage 3 trains the defense model. The model takes the ten multi-noise embeddings, learns a softmax weighting over noise levels, passes the weighted embedding through an MLP, and trains with ArcFace loss on CelebA identities.

Stage 4 attacks the defended system directly, using defended embeddings as the IP-Adapter input.

Stage 5 trains a LoRA adapter against defended embeddings and then attacks/evaluates with that adapted generator. The current run configuration uses raw CelebA images rather than cached VAE latents for LoRA training, `batch_size=1`, `lr=5e-5`, `20` epochs, validation-loss checkpoint selection, per-step loss logging, and a smoothed loss curve.

## Directory Layout

Expected project layout:

```text
facereco/
  code/
    facereco/
    run_pipeline.py
    run.sh
    job.sh
    visualize_tsne.py
    requirements.txt
  data/
    celeba/
      img_align_celeba/
      list_eval_partition.csv
      identity_CelebA.txt
      list_landmarks_align_celeba.txt
  models/
    realvisxl-v3.0/
    IP-Adapter/
    ip-adapter-faceid/
      ip-adapter-faceid_sdxl.bin
  work/
    embeddings/
    checkpoints/
    generated/
    metrics/
    comparisons/
    visualizations/
```

The code assumes the repository root is one level above `code/`. Run commands from `code/` unless you explicitly pass `--repo-root`, `--data-dir`, `--work-dir`, and `--model-dir`.

## Data Requirements

Place CelebA under `../data/celeba`.

Required:

- `img_align_celeba/`: CelebA aligned images.
- `list_eval_partition.csv`: official CelebA train/val/test split.
- `identity_CelebA.txt` or an equivalent identity file. The scripts also accept common variants such as `identity_CelebA.csv`, `identity_celeba.txt`, and `list_identity_celeba.txt`.

Recommended for fast Stage 0:

- `list_landmarks_align_celeba.txt`: official CelebA landmarks. With this file, Stage 0 can use `--embedding-mode landmarks`, which avoids running a detector for every image.

## Model Requirements

Required local models:

- RealVisXL / SDXL base model at `../models/realvisxl-v3.0`.
- IP-Adapter source repository at `../models/IP-Adapter`.
- FaceID XL checkpoint at `../models/ip-adapter-faceid/ip-adapter-faceid_sdxl.bin`.
- InsightFace buffalo_l. InsightFace usually downloads this into `~/.insightface/models/buffalo_l` on first use.

`run.sh` can clone IP-Adapter and download `ip-adapter-faceid_sdxl.bin` if they are missing. It does not download CelebA or RealVisXL.

## Environment Requirements

Recommended hardware:

- Linux GPU node with one CUDA-capable NVIDIA GPU.
- CUDA 12.x runtime.
- At least 32 GB CPU memory.
- Enough disk space for CelebA, embeddings, latents, generated images, and checkpoints.

The provided Slurm job requests:

```bash
#SBATCH -p emergency_gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
```

The current `run.sh` expects a conda environment named `face` and initializes it with:

```bash
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate face
```

If your cluster uses a different conda path or environment name, edit the first lines of `run.sh`.

## Installation

From `code/`, install Python dependencies into the `face` environment:

```bash
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate face
python -m pip install -r requirements.txt
```

`run.sh` also contains defensive checks that install compatible versions of PyTorch, diffusers, PEFT, and ONNXRuntime GPU if needed. For reproducibility, prefer installing `requirements.txt` first.

After installation, verify CUDA:

```bash
python - <<'PY'
import torch
import onnxruntime as ort
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("onnxruntime providers", ort.get_available_providers())
PY
```

`CUDAExecutionProvider` must be present for buffalo_l. If InsightFace silently falls back to CPU, Stage 0 will be extremely slow.

## Step-by-Step Usage

### 1. Create the Official Split Manifest

```bash
python run_pipeline.py split
```

Output:

```text
../work/celeba_official_split.csv
```

### 2. Stage 0: Cache Multi-Noise buffalo_l Embeddings

Fast landmark-based mode:

```bash
python run_pipeline.py stage0 \
  --batch-size 512 \
  --ctx-id 0 \
  --det-size 320 \
  --save-every 8 \
  --embedding-mode landmarks \
  --recognition-batch-size 2048 \
  --preprocess-workers 1
```

This writes sharded caches under:

```text
../work/embeddings/noise_00/
../work/embeddings/noise_10/
...
../work/embeddings/noise_90/
```

Then export the clean 0% embedding table:

```bash
python run_pipeline.py export-level --level 0
```

Output:

```text
../work/embeddings/buffalo_l_noise_00.pt
```

Optional SDXL VAE latent cache:

```bash
python run_pipeline.py stage0 \
  --skip-embeddings \
  --cache-vae \
  --vae-batch-size 8 \
  --vae-image-size 512 \
  --device cuda
```

The current Stage 5 configuration does not use cached latents by default, but the option is still available with `--use-cached-latents`.

### 3. Stage 1: Original Retrieval

```bash
python run_pipeline.py stage1 \
  --query-split test \
  --gallery-split test \
  --level 0
```

Output:

```text
../work/metrics/stage1_retrieval.json
```

CelebA official train/val/test identities are not ideal for cross-split retrieval because identities may not overlap. The default evaluation uses test-vs-test and excludes self matches.

### 4. Stage 2: Attack Original Embeddings

Generate reconstructions for the first 200 test images:

```bash
python run_pipeline.py stage2 \
  --embedding-file ../work/embeddings/buffalo_l_noise_00.pt \
  --generated-dir ../work/generated/stage2_test200_768x1024 \
  --limit 200 \
  --width 768 \
  --height 1024 \
  --prompt "A centered realistic head-and-shoulders portrait photo of one person, full face fully visible, entire head visible, natural lighting, realistic skin texture" \
  --negative-prompt "cropped face, partial face, cut off head, out of frame, extreme closeup, monochrome, lowres, bad anatomy, worst quality, low quality, blurry" \
  --generate \
  --evaluate
```

Outputs:

```text
../work/generated/stage2_test200_768x1024/
../work/metrics/stage2_reconstruction.json
```

### 5. Stage 3: Train the Defense Model

```bash
python run_pipeline.py stage3 \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-3
```

Model details:

- Input: ten buffalo_l embeddings from noise levels `0` through `90`.
- Noise mixing: trainable softmax weights.
- MLP: `512 -> 384 -> 512`, batch norm, PReLU, dropout `0.8`.
- Loss: ArcFace with scale `64.0` and margin `0.5`.
- Optimizer: AdamW, weight decay `1e-4`.

Outputs:

```text
../work/checkpoints/defense_arcface.pt
../work/checkpoints/defense_arcface.json
```

Export defended embeddings:

```bash
python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split train
python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split val
python run_pipeline.py export-defended --ckpt ../work/checkpoints/defense_arcface.pt --split test
```

Outputs:

```text
../work/embeddings/defended_train.pt
../work/embeddings/defended_val.pt
../work/embeddings/defended_test.pt
```

### 6. Stage 4: Attack Defended Embeddings Directly

```bash
python run_pipeline.py stage4 \
  --embedding-file ../work/embeddings/defended_test.pt \
  --generated-dir ../work/generated/stage4_test200_768x1024 \
  --limit 200 \
  --width 768 \
  --height 1024 \
  --prompt "A centered realistic head-and-shoulders portrait photo of one person, full face fully visible, entire head visible, natural lighting, realistic skin texture" \
  --negative-prompt "cropped face, partial face, cut off head, out of frame, extreme closeup, monochrome, lowres, bad anatomy, worst quality, low quality, blurry" \
  --generate \
  --evaluate
```

Outputs:

```text
../work/generated/stage4_test200_768x1024/
../work/metrics/stage4_reconstruction.json
```

### 7. Stage 5: Train LoRA Against Defended Embeddings and Evaluate

Current raw-image Stage 5 configuration:

```bash
python run_pipeline.py stage5 \
  --embedding-file ../work/embeddings/defended_train.pt \
  --val-embedding-file ../work/embeddings/defended_val.pt \
  --eval-embedding-file ../work/embeddings/defended_test.pt \
  --lora-dir ../work/checkpoints/ip_adapter_defended_lora_rawimg_bs1_lr5e5_ep20 \
  --generated-dir ../work/generated/stage5_test200_768x1024_rawimg_bs1_lr5e5_ep20 \
  --metrics-out ../work/metrics/stage5_reconstruction_rawimg_bs1_lr5e5_ep20.json \
  --limit 200 \
  --width 768 \
  --height 1024 \
  --prompt "A centered realistic head-and-shoulders portrait photo of one person, full face fully visible, entire head visible, natural lighting, realistic skin texture" \
  --negative-prompt "cropped face, partial face, cut off head, out of frame, extreme closeup, monochrome, lowres, bad anatomy, worst quality, low quality, blurry" \
  --epochs 20 \
  --steps-per-epoch 1000 \
  --val-batches 50 \
  --batch-size 1 \
  --lr 5e-5 \
  --max-grad-norm 1.0 \
  --loss-smooth-window 100 \
  --generate \
  --evaluate
```

Stage 5 training details:

- Training pairs: raw CelebA train images and defended train embeddings.
- Image preprocessing: resize `512`, center crop `512`, normalize to `[-1, 1]`.
- Latents: computed online with the SDXL VAE for each batch.
- Batch size: `1`.
- Epochs: `20`.
- Steps per epoch: `1000`.
- Validation: first `50` validation batches after each epoch.
- Checkpoint selection: lowest validation loss.
- Step loss logging: every training step is appended to `lora_step_losses.jsonl`.
- Curve plotting: `lora_loss_curve.png` is drawn from step-level loss data with smoothing.

Outputs:

```text
../work/checkpoints/ip_adapter_defended_lora_rawimg_bs1_lr5e5_ep20/adapter_model.safetensors
../work/checkpoints/ip_adapter_defended_lora_rawimg_bs1_lr5e5_ep20/lora_step_losses.jsonl
../work/checkpoints/ip_adapter_defended_lora_rawimg_bs1_lr5e5_ep20/lora_loss_history.json
../work/checkpoints/ip_adapter_defended_lora_rawimg_bs1_lr5e5_ep20/lora_loss_curve.png
../work/generated/stage5_test200_768x1024_rawimg_bs1_lr5e5_ep20/
../work/metrics/stage5_reconstruction_rawimg_bs1_lr5e5_ep20.json
```

To train Stage 5 from cached VAE latents instead of raw images, add:

```bash
--use-cached-latents
```

To preload those cached latents into GPU memory, keep the default behavior. To disable GPU preload, add:

```bash
--no-preload-latents-to-gpu
```

## Full Slurm Run

The current `run.sh` is intentionally focused on the already-completed pipeline state: it verifies the environment, writes the split manifest, exports level-0 embeddings, runs Stage 5, evaluates Stage 5, and creates a comparison sheet if Stage 2 and Stage 4 generated folders already exist.

Submit from `code/`:

```bash
sbatch job.sh
```

`job.sh` runs:

```bash
bash run.sh
```

Monitor:

```bash
squeue -u "$USER"
tail -f job_<JOBID>.out
tail -f job_<JOBID>.err
```

## Reconstruction Metrics

Reconstruction evaluation reports:

- `insightface_cosine_mean`: cosine similarity between buffalo_l embeddings of the ground-truth and generated image.
- `clip_vitl14_cosine_mean`: cosine similarity between CLIP image features of the ground-truth and generated image.

CLIP model:

```text
openai/clip-vit-large-patch14
```

Implementation:

```python
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
```

## Comparison Sheets

Create a GT / Stage 2 / Stage 4 / Stage 5 comparison sheet:

```bash
python run_pipeline.py compare \
  --stage2-dir ../work/generated/stage2_test200_768x1024 \
  --stage4-dir ../work/generated/stage4_test200_768x1024 \
  --stage5-dir ../work/generated/stage5_test200_768x1024_rawimg_bs1_lr5e5_ep20 \
  --out ../work/comparisons/gt_stage2_stage4_stage5_rawimg_bs1_lr5e5_ep20.jpg \
  --limit 8
```

## t-SNE Visualization

Draw a 2D visualization for 200 test embeddings before and after defense:

```bash
python visualize_tsne.py \
  --before ../work/embeddings/buffalo_l_noise_00.pt \
  --after ../work/embeddings/defended_test.pt \
  --out ../work/visualizations/defense_tsne_test200.png \
  --limit 200 \
  --backend torch
```

The script uses scikit-learn t-SNE if available. If scikit-learn is missing or incompatible, the `torch` backend can still generate a t-SNE-style visualization without matplotlib.

## Common Problems

If buffalo_l is slow, check ONNXRuntime providers. It must include `CUDAExecutionProvider`.

If `scikit-learn` or `matplotlib` fails with NumPy ABI errors, use `requirements.txt`, which pins `numpy==1.26.4`. The visualization script does not require matplotlib.

If Stage 5 loss is unstable, check the current `--batch-size`, `--lr`, and whether training is using raw images or cached latents. The current raw-image setting is `batch_size=1` and `lr=5e-5`.

If IP-Adapter import fails, make sure `../models/IP-Adapter` exists and that `PYTHONPATH` includes it. `run.sh` handles this automatically.

If generated images are skipped unexpectedly, check whether the target generated directory already contains images with matching filenames. The generation code skips existing outputs.
