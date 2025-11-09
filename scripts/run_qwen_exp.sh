#!/bin/bash
#SBATCH --job-name=qwen_ppm
#SBATCH --output=/home/20201100/sem-repl/logs/qwen_%A_%a.out
#SBATCH --error=/home/20201100/sem-repl/logs/qwen_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:l4.22gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35

# Create logs directory if it doesn't exist
mkdir -p /home/20201100/sem-repl/logs

# Load required modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.1.1

# Activate virtual environment
source /home/20201100/sem-repl/venv/bin/activate

# Change to project directory
cd /home/20201100/sem-repl

# Read parameters from file
PARAMS_FILE="/home/20201100/sem-repl/scripts/qwen_params.txt"
PARAM_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $PARAMS_FILE)

# Print job info
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Parameters: $PARAM_LINE"
echo "=================================================="

# Run the experiment
python next_event_prediction.py $PARAM_LINE

# Check exit status
if [ $? -eq 0 ]; then
    echo "Experiment $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "Experiment $SLURM_ARRAY_TASK_ID failed with exit code $?"
fi