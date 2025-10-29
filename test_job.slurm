#!/bin/bash
#SBATCH --job-name=test_ppm
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err


# Setup
module load Python/3.12.3-GCCcore-13.3.0
source /home/20201100/llm-ppm-replication/llm-peft-ppm/venv/bin/activate
mkdir -p logs

# Print environment info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU info:"
nvidia-smi

# Run a minimal test - RNN baseline with 1 epoch
python next_event_prediction.py \
    --dataset BPI20PrepaidTravelCosts \
    --backbone rnn \
    --embedding_size 32 \
    --hidden_size 128 \
    --lr 0.0005 \
    --batch_size 64 \
    --epochs 1 \
    --categorical_features activity \
    --continuous_features all \
    --categorical_targets activity \
    --continuous_targets remaining_time

echo "Job finished at: $(date)"