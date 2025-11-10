# LLM-PEFT-PPM Replication Study

**Replication of: Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring**

Original paper: Oyamada et al. "Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring" (arXiv 2025)

Original code repository: [https://github.com/raseidi/llm-peft-ppm](https://github.com/raseidi/llm-peft-ppm)

## Abstract

I have attempted to replicate the methods and reproduce the results in "Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring" using publicly available datasets and the authors' published code. 

The original study demonstrated that PEFT-adapted LLMs outperform traditional RNN-based approaches and narrative-style LLM methods in both single-task and multi-task predictive process monitoring scenarios. I replicated their experimental framework using identical model architectures (LSTM, GPT-2, Qwen2, Llama3.2), PEFT techniques (LoRA with r=256, α=512, and layer freezing strategies), and evaluation metrics (accuracy for next activity prediction, MSE for remaining time prediction).

I extended the original evaluation by including the BPI Traffic Fines dataset to assess the generalizability of the proposed approach across a broader range of process types and characteristics. My replication closely follows the original methodology with identical preprocessing steps, train/test splitting, and hyperparameter configurations. I ran the experiments 10 times to account for the variability and reported average performance metrics along with standard deviations.


## Requirements

### Software Requirements
- Python >= 3.12
- PyTorch >= 2.5.1  
- Transformers >= 4.0
- PEFT >= 0.3.0
- Datasets >= 2.0
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- pyyaml
- pm4py==2.7.19

### HPC Environment
This replication was conducted on:
- **Platform**: TU/e Umbrella HPC Cluster
- **GPU**: NVIDIA GPUs with CUDA 11.7+
- **Environment**: Jupyter Lab 3.5.0 with PyTorch 1.x support
- **Job Scheduler**: SLURM

## Installation and Setup

### 1. Environment Setup
```bash
# Clone this replication repository
git clone [this-repo-url]
cd llm-peft-ppm-replication

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Original Code Setup
The original code is automatically cloned and configured when running the setup notebook:
```bash
# Original repository will be cloned to: ./original_repo/
git clone https://github.com/raseidi/llm-peft-ppm.git original_repo
```

### 3. HuggingFace Configuration
```bash
# Obtain HuggingFace token from: https://huggingface.co/settings/tokens
# Set token (choose one method):

# Method 1: Environment variable
export HF_TOKEN="your_huggingface_token_here"

# Method 2: .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Method 3: Configure in notebook during setup
```

## Datasets

### Original Datasets (Automatically Downloaded)
The following datasets are automatically downloaded via SkPM during experiments:

1. **BPI Challenge 2012 (BPI12)**: Loan application process
2. **BPI Challenge 2017 (BPI17)**: Credit application process  
3. **BPI Challenge 2020 - Request for Payment (BPI20RfP)**
4. **BPI Challenge 2020 - Prepaid Travel Costs (BPI20PTC)**
5. **BPI Challenge 2020 - Permit Data (BPI20PD)**
### Extension Dataset needs to be downloaded manually
6. **BPI Traffic Fines**: Municipal traffic violation process

## Preprocessing
Preprocessing is handled automatically when running experiments (even for the BPI Traffic Fines dataset). 


## Experimental Framework

### Model Architectures

#### Baseline Models
- **LSTM Networks (RNN)**: Traditional sequence modeling baseline
  - Configurations: Single-task (ST-RNN) and Multi-task (MT-RNN)
  - Hyperparameter search across: layers (1-6), learning rates (5e-4, 1e-4, 5e-5), embedding dimensions (32, 128, 256, 512), hidden dimensions (128, 256, 512), batch sizes (32, 64, 256)

#### LLM Baselines  
- **PM-GPT2**: GPT-2 adapted for process data following transfer learning principles
- **S-NAP**: Narrative-style approach using Llama with LoRA fine-tuning on text-converted process data

#### Proposed PEFT Models
- **Qwen2** (0.5B parameters)
- **Llama3.2** (1B parameters)  
- **PM-GPT2** (0.1B parameters)

### PEFT Configurations

#### 1. LoRA Adaptation
```bash
--fine_tuning lora --r 256 --lora_alpha 512
```

#### 2. Layer Freezing Strategies
- **Full Freezing**: All backbone parameters frozen
- **Partial Freezing**: Selective unfreezing of specific layers
  - First layers: `--freeze_layers 0` or `--freeze_layers 0,1`  
  - Last layers: `--freeze_layers -1` or `--freeze_layers -1,-2`

### Training Configuration
- **LLM Fine-tuning**: 10 epochs
- **RNN Training**: 25 epochs  
- **Loss Functions**: Cross-entropy (next activity), MSE (remaining time)
- **Optimization**: Grid search for systematic hyperparameter evaluation
- **Multi-task Setup**: Single model predicting both next activity and remaining time

## Running Experiments
Scripts for running the experiments are provided in the `scripts/` directory. They are of the format `run_[model]_exp.sh`.
On the HPC cluster, use SLURM job scripts to submit experiments.

## Evaluation

### Metrics
- **Next Activity Prediction**: Classification accuracy
- **Remaining Time Prediction**: Mean Squared Error (MSE)  
- **Convergence Analysis**: Loss curves and training stability

### Evaluation Scripts


## Reproducibility

### Computational Environment
- **Hardware**: TU/e HPC with NVIDIA GPUs
- **Software**: PyTorch 2.5.1, CUDA 11.7+
- **Random Seeds**: Fixed for reproducibility

### Data Availability  
- All datasets are publicly available through SkPM
- Preprocessing scripts ensure consistent data preparation
- Train/test splits follow original methodology exactly


## Citation

Original paper citation:
```bibtex
@article{oyamada2024peft,
  title={Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring},
  author={Oyamada, Rafael and [Other Authors]},
  journal={[Journal Name]},
  year={2024},
  note={Under Review}
}
```

## Acknowledgments

- **Original Authors**: Rafael Oyamada et al. for providing the original implementation
- **TU/e HPC**: For computational resources and infrastructure support  
- **Datasets**: BPI Challenge organizers for providing publicly available process mining datasets
- **Libraries**: HuggingFace Transformers, PyTorch, and PEFT library developers


## Contact

For questions about this replication study:
- **Student**: Christine Jacob - [c.christine.jacob@student.tue.nl]
- **Institution**: Eindhoven University of Technology
- **Course**: Seminar Process Analytics (2025)

For questions about the original method:
- **Original Author**: Rafael Oyamada - [rafael.oyamada@kuleuven.be]
- **Original Repository**: [https://github.com/raseidi/llm-peft-ppm](https://github.com/raseidi/llm-peft-ppm)


