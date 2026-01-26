#!/bin/bash
#SBATCH --job-name=nnssl_preprocess
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00\
#SBATCH --nodelist=rri-cbs-h7.schulich.uwo.ca

cd /nfs/khan/trainees/apooladi/abeta/nnssl
pixi run nnssl_preprocess -d 1 -part_id $SLURM_ARRAY_TASK_ID -num_parts 20
