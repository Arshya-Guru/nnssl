#!/bin/bash
# init16_fast.sh - FAST training with lazy caching
#
# Training starts IMMEDIATELY - no waiting for data transfer!
# Files are cached to /tmp on-demand as they're accessed.
#
# SETUP (one-time):
#   1. Copy caching_dataset.py to src/nnssl/data/dataloading/caching_dataset.py
#   2. Copy run_cached.py to the nnssl repo root
#
# USAGE:
#   salloc --gres=gpu:l40s:2 --cpus-per-task=128 --mem=256G --time=24:00:00
#   cd /nfs/khan/trainees/apooladi/abeta/nnssl
#   source init16_fast.sh
#   run_training 1 noresample -tr SimCLRTrainer_BS64 -p nnsslPlans -num_gpus 2

set -e

# =============================================================================
# CONFIGURATION - Edit these paths as needed
# =============================================================================
NFS_RAW="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/raw"
NFS_PREPROCESSED="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/preprocessed"
NFS_RESULTS="/nfs/khan/trainees/apooladi/abeta/nnssl_data/16/nnssl_data/results"

# Local fast storage
JOBID="${SLURM_JOB_ID:-$$}"
LOCAL_BASE="/tmp/${USER}/nnssl_${JOBID}"
LOCAL_CACHE="${LOCAL_BASE}/cache"
LOCAL_RESULTS="${LOCAL_BASE}/results"

# =============================================================================
# SETUP
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           nnssl Fast Training with Lazy Caching                  ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Training starts IMMEDIATELY - files cached on-demand to /tmp   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

mkdir -p "$LOCAL_CACHE" "$LOCAL_RESULTS"

# Export for the Python caching module
export NNSSL_NFS_PREPROCESSED="$NFS_PREPROCESSED"
export NNSSL_LOCAL_CACHE="$LOCAL_CACHE"
export NNSSL_NFS_RESULTS="$NFS_RESULTS"
export NNSSL_LOCAL_RESULTS="$LOCAL_RESULTS"

# Standard nnssl paths - point to NFS for metadata, local for results
export nnssl_raw="$NFS_RAW"
export nnssl_preprocessed="$NFS_PREPROCESSED"
export nnssl_results="$LOCAL_RESULTS"

# CPU workers for data augmentation (tune based on your allocation)
# With 64 CPUs per GPU, 32 workers/GPU is a good balance
NUM_GPUS="${NUM_GPUS:-2}"
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-128}"
WORKERS_PER_GPU=$(( (TOTAL_CPUS / NUM_GPUS) - 4 ))  # Leave some for main process
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-$WORKERS_PER_GPU}"

echo "Configuration:"
echo "  NFS preprocessed: $NFS_PREPROCESSED"
echo "  Local cache:      $LOCAL_CACHE"
echo "  Local results:    $LOCAL_RESULTS"
echo "  Workers/GPU:      $nnUNet_n_proc_DA"
echo "  Job ID:           $JOBID"
echo ""

# =============================================================================
# VERIFY SETUP
# =============================================================================
NNSSL_DIR="$(pwd)"
CACHED_SCRIPT="$NNSSL_DIR/run_cached.py"
CACHING_MODULE="$NNSSL_DIR/src/nnssl/data/dataloading/caching_dataset.py"

if [ ! -f "$CACHED_SCRIPT" ]; then
    echo "⚠ Warning: run_cached.py not found in $NNSSL_DIR"
    echo "  Copy it here: cp /path/to/run_cached.py $NNSSL_DIR/"
fi

if [ ! -f "$CACHING_MODULE" ]; then
    echo "⚠ Warning: caching_dataset.py not found"
    echo "  Copy it here: cp /path/to/caching_dataset.py $CACHING_MODULE"
fi

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

sync_results() {
    echo ""
    echo "[$(date +%H:%M:%S)] Syncing results to NFS..."
    rsync -av "$LOCAL_RESULTS/" "$NFS_RESULTS/" 2>/dev/null || cp -r "$LOCAL_RESULTS/"* "$NFS_RESULTS/" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Done! Results at: $NFS_RESULTS"
}

cache_status() {
    if [ -d "$LOCAL_CACHE" ]; then
        local size=$(du -sh "$LOCAL_CACHE" 2>/dev/null | cut -f1)
        local files=$(find "$LOCAL_CACHE" -type f 2>/dev/null | wc -l)
        echo "Cache: $size ($files files)"
    fi
}

cleanup() {
    echo "Cleaning up local storage..."
    rm -rf "$LOCAL_BASE"
}

export -f sync_results cache_status cleanup
export NFS_RESULTS LOCAL_RESULTS LOCAL_CACHE NNSSL_DIR

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

run_training() {
    local args="$@"
    
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Starting training with lazy caching                             ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Command: nnssl_train $args"
    echo ""
    
    # Use the actual Python script file (not heredoc/stdin)
    # This is required for PyTorch multiprocessing spawn to work
    pixi run python "$NNSSL_DIR/run_cached.py" $args
    
    local exit_code=$?
    
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    cache_status
    sync_results
    
    return $exit_code
}

export -f run_training

# =============================================================================
# READY MESSAGE
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Ready! Use 'run_training' to start:                             ║"
echo "║                                                                  ║"
echo "║  run_training 1 noresample -tr SimCLRTrainer_BS64 \\              ║"
echo "║               -p nnsslPlans -num_gpus 2                          ║"
echo "║                                                                  ║"
echo "║  Utilities:                                                      ║"
echo "║    cache_status  - Show cache size                               ║"
echo "║    sync_results  - Manually sync results to NFS                  ║"
echo "║    cleanup       - Remove local cache                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
