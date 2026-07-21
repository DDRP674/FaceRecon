#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH --time=03:00:00
#SBATCH -o job_%j.out
#SBATCH --gres=gpu:1
#SBATCH -e job_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

bash run.sh
