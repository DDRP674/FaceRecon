#!/bin/bash
#SBATCH -p emergency_cpu
#SBATCH --time=01:00:00
#SBATCH -o tsne_%j.out
#SBATCH -e tsne_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

set +u
source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh
conda activate face
set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python visualize_tsne.py \
  --before ../work/embeddings/buffalo_l_noise_00.pt \
  --after ../work/embeddings/defended_test.pt \
  --out ../work/visualizations/defense_tsne_test200.png \
  --limit 200 \
  --perplexity 30 \
  --backend torch \
  --device cpu \
  --iterations 300 \
  --lr 10
