#!/bin/bash
#SBATCH --job-name=nnssl_preprocess_3_4
#SBATCH --array=3-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --nodelist=rri-cbs-h7.schulich.uwo.ca

set -euo pipefail

export nnssl_raw="/localscratch/apooladi/nnssl_data/raw"
export nnssl_preprocessed="/localscratch/apooladi/nnssl_data/preprocessed"
export nnssl_results="/localscratch/apooladi/nnssl_data/results"

cd /nfs/khan/trainees/apooladi/abeta/nnssl

# Keep -np aligned with cpus-per-task to avoid oversubscription.
# 2 is a safe middle-ground: parallelism without huge RAM spikes.
pixi run nnssl_preprocess -d 1 -c noresample \
  -part $SLURM_ARRAY_TASK_ID -total_parts 15 \
  -np 2 --verbose
