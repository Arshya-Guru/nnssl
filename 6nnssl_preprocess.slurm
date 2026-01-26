#!/bin/bash
#SBATCH --job-name=nnssl_preprocess
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=rri-cbs-h6.schulich.uwo.ca

export nnssl_raw="/localscratch/apooladi/nnssl_data/raw"
export nnssl_preprocessed="/localscratch/apooladi/nnssl_data/preprocessed"
export nnssl_results="/localscratch/apooladi/nnssl_data/results"
cd /nfs/khan/trainees/apooladi/abeta/nnssl
pixi run nnssl_preprocess -d 1 -c noresample -part_id $SLURM_ARRAY_TASK_ID -num_parts 15 -np 4 
