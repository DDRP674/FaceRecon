#!/bin/bash
#SBATCH -p emergency_gpu
#SBATCH --time=12:00:00
#SBATCH -o job_%j.out
#SBATCH --gres=gpu:1
#SBATCH -e job_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

bash run.sh
