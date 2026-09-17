#!/usr/bin/env bash
#SBATCH --job-name=aim3-atari-quota-probe
#SBATCH --partition=gpu
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=adalovelace
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail

# Read the scratch quota from a GPU compute node; login-node accounting may differ.
mmlsquota -u "${QUOTA_USER:-$USER}" scratch
df -h /scratch
