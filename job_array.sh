#!/bin/bash
#SBATCH --job-name=polyfem_cell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --array=0-19
#SBATCH --output=logs/cell_%A_%a.out
#SBATCH --error=logs/cell_%A_%a.err

set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.12.4
source /home/teseo/scratch/polyfem-env/bin/activate

BASE=/home/teseo/scratch/cell
POLYFEM=/home/teseo/scratch/polyfem/build/PolyFEM_bin

mkdir -p "$BASE/logs"

RUNS_PER_JOB=12
START=$((SLURM_ARRAY_TASK_ID * RUNS_PER_JOB))
END=$((START + RUNS_PER_JOB - 1))

if [ "$END" -gt 221 ]; then
    END=221
fi

echo "Array task $SLURM_ARRAY_TASK_ID running indices $START to $END"

for INDEX in $(seq "$START" "$END"); do
    RUN_DIR="$BASE/runs/index_${INDEX}"
    mkdir -p "$RUN_DIR"

    echo "Running index $INDEX in $RUN_DIR"

    export JOB_INDEX=$INDEX

    cd "$RUN_DIR"
    "$POLYFEM" -j "$BASE/run.json" -o "$RUN_DIR/outnz"
done