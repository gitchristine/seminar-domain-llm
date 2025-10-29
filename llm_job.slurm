#!/bin/bash
#SBATCH --job-name=llm_multi
#SBATCH --partition=tue.gpu.q
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/llm_multi_%j.out
#SBATCH --error=logs/llm_multi_%j.err

module load Python/3.12.3-GCCcore-13.3.0
source /home/20201100/llm-ppm-replication/llm-peft-ppm/venv/bin/activate
mkdir -p logs

echo "========================================"
echo "Multi-LLM Experiment Suite"
echo "Job started at: $(date)"
echo "========================================"
nvidia-smi

# Experiment 1: Qwen with LoRA
echo ""
echo "=== Experiment 1: Qwen + LoRA ==="
python next_event_prediction.py --dataset BPI20PrepaidTravelCosts --backbone qwen25-05b --embedding_size 896 --hidden_size 896 --lr 0.00005 --batch_size 8 --epochs 10 --categorical_features activity --continuous_features all --categorical_targets activity --continuous_targets remaining_time --fine_tuning lora --r 256 --lora_alpha 512

echo "Experiment 1 completed at: $(date)"

# Experiment 2: Qwen with layer freezing
echo ""
echo "=== Experiment 2: Qwen + Freeze Layers ==="
python next_event_prediction.py --dataset BPI20PrepaidTravelCosts --backbone qwen25-05b --embedding_size 896 --hidden_size 896 --lr 0.00005 --batch_size 8 --epochs 10 --categorical_features activity --continuous_features all --categorical_targets activity --continuous_targets remaining_time --fine_tuning freeze --freeze_layers -1 -2

echo "Experiment 2 completed at: $(date)"

# Experiment 3: PM-GPT2 with LoRA
echo ""
echo "=== Experiment 3: PM-GPT2 + LoRA ==="
python next_event_prediction.py --dataset BPI20PrepaidTravelCosts --backbone pm-gpt2 --embedding_size 768 --hidden_size 768 --lr 0.00005 --batch_size 8 --epochs 10 --categorical_features activity --continuous_features all --categorical_targets activity --continuous_targets remaining_time --fine_tuning lora --r 256 --lora_alpha 512

echo "Experiment 3 completed at: $(date)"
echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"