#!/bin/bash
#SBATCH --job-name=nnssl_train
#SBATCH --output=logs/nnssl_%j.out
#SBATCH --error=logs/nnssl_%j.err
#SBATCH --gres=gpu:2
#SBATCH --nodelist=rri-cbs-h1.schulich.uwo.ca
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --signal=SIGUSR1@300

# =============================================================================
# nnssl_staged_train.sbatch
# SLURM batch script for nnssl training with local /tmp staging
#
# Usage:
#   sbatch nnssl_staged_train.sbatch <trainer_name> [additional_args]
#
# Examples:
#   sbatch nnssl_staged_train.sbatch SimCLRTrainer_BS64
#   sbatch nnssl_staged_train.sbatch BYOLTrainer_BS8_256iso
#
# Or modify the TRAINER variable below and submit without args.
# =============================================================================

# Default trainer (can be overridden by command line arg)
TRAINER="${1:-SimCLRTrainer_BS64}"
shift 2>/dev/null || true  # Shift past first arg if provided

# Dataset configuration
DATASET_ID="1"
CONFIG="noresample"
PLANS="nnsslPlans"
NUM_GPUS=2

# Additional args passed to script
EXTRA_ARGS="$@"

# =============================================================================
# PATHS
# =============================================================================
NFS_RAW="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/raw"
NFS_PREPROCESSED="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/preprocessed"
NFS_RESULTS="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/results"

LOCAL_BASE="/tmp/${USER}/nnssl_${SLURM_JOB_ID}"

# =============================================================================
# CREATE LOG DIRECTORY
# =============================================================================
mkdir -p logs

# =============================================================================
# PRINT JOB INFO
# =============================================================================
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on:   $(hostname)"
echo "Date:         $(date)"
echo "=========================================="
echo "Configuration:"
echo "  Trainer:    $TRAINER"
echo "  Dataset:    $DATASET_ID"
echo "  Config:     $CONFIG"
echo "  Plans:      $PLANS"
echo "  GPUs:       $NUM_GPUS"
echo "  CPUs:       $SLURM_CPUS_PER_TASK"
echo "  Extra args: $EXTRA_ARGS"
echo "=========================================="

# =============================================================================
# SETUP STAGING
# =============================================================================
echo ""
echo "Setting up local staging at: $LOCAL_BASE"
mkdir -p "$LOCAL_BASE/raw"
mkdir -p "$LOCAL_BASE/preprocessed"
mkdir -p "$LOCAL_BASE/results"

# Function to sync results back
sync_results() {
    echo ""
    echo "[$(date)] Syncing results to NFS..."
    rsync -av "$LOCAL_BASE/results/" "$NFS_RESULTS/"
    echo "[$(date)] Results synced!"
}

# Function to handle job termination signals
handle_signal() {
    echo ""
    echo "[$(date)] Received termination signal, syncing results..."
    sync_results
    exit 0
}

# Register signal handlers
trap handle_signal SIGTERM SIGINT SIGUSR1

# =============================================================================
# STAGE DATA TO LOCAL
# =============================================================================
echo ""
echo "[$(date)] Staging preprocessed data..."
rsync -a --info=progress2 "$NFS_PREPROCESSED/" "$LOCAL_BASE/preprocessed/"

echo "[$(date)] Staging raw data..."
rsync -a --info=progress2 "$NFS_RAW/" "$LOCAL_BASE/raw/"

echo "[$(date)] Data staging complete!"

# =============================================================================
# SET ENVIRONMENT
# =============================================================================
export nnssl_raw="$LOCAL_BASE/raw"
export nnssl_preprocessed="$LOCAL_BASE/preprocessed"
export nnssl_results="$LOCAL_BASE/results"

# Calculate workers per GPU (leave some cores for system/main process)
WORKERS_PER_GPU=$(( (SLURM_CPUS_PER_TASK / NUM_GPUS) - 4 ))
export nnUNet_n_proc_DA="$WORKERS_PER_GPU"

echo ""
echo "Environment:"
echo "  nnssl_raw=$nnssl_raw"
echo "  nnssl_preprocessed=$nnssl_preprocessed"
echo "  nnssl_results=$nnssl_results"
echo "  nnUNet_n_proc_DA=$nnUNet_n_proc_DA"

# =============================================================================
# ACTIVATE ENVIRONMENT AND RUN TRAINING
# =============================================================================
echo ""
echo "[$(date)] Starting training..."
echo "Command: pixi run nnssl_train $DATASET_ID $CONFIG -tr $TRAINER -p $PLANS -num_gpus $NUM_GPUS $EXTRA_ARGS"
echo ""

# Change to your nnssl directory
cd /nfs/khan/trainees/apooladi/abeta/nnssl

# Run training
pixi run nnssl_train "$DATASET_ID" "$CONFIG" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -num_gpus "$NUM_GPUS" \
    $EXTRA_ARGS

TRAIN_EXIT_CODE=$?

# =============================================================================
# SYNC RESULTS AND CLEANUP
# =============================================================================
echo ""
echo "[$(date)] Training finished with exit code: $TRAIN_EXIT_CODE"
sync_results

# Optional: cleanup local staging (comment out if you want to inspect)
# rm -rf "$LOCAL_BASE"

echo "[$(date)] Job complete!"
exit $TRAIN_EXIT_CODE
