# LLM-PEFT-PPM Replication Study

**Replication of: Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring**

Original paper: Oyamada et al. "Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring" (Under Review)

Original code repository: [https://github.com/raseidi/llm-peft-ppm](https://github.com/raseidi/llm-peft-ppm)

## Abstract

We have attempted to replicate the methods and reproduce the results in "Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring" using publicly available datasets and the authors' published code. This replication study was conducted on TU/e HPC infrastructure to evaluate the effectiveness of PEFT techniques for adapting Large Language Models to predictive process monitoring tasks.

The original study demonstrated that PEFT-adapted LLMs outperform traditional RNN-based approaches and narrative-style LLM methods in both single-task and multi-task predictive process monitoring scenarios. We replicated their experimental framework using identical model architectures (LSTM, GPT-2, Qwen2, Llama3.2), PEFT techniques (LoRA with r=256, α=512, and layer freezing strategies), and evaluation metrics (accuracy for next activity prediction, MSE for remaining time prediction).

We extended the original evaluation by including the BPI Traffic Fines dataset to assess the generalizability of the proposed approach across a broader range of process types and characteristics. Our replication closely follows the original methodology with identical preprocessing steps, train/test splitting, and hyperparameter configurations.

[Results summary to be added upon completion of experiments]

This study demonstrates the [challenges/successes] of reproducing deep learning method results in the process mining domain, and contributes to the validation of PEFT techniques for predictive process monitoring applications.

## Requirements

### Python Requirements
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

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- 32GB+ RAM recommended for larger datasets
- High-speed storage for dataset processing

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
git clone [your-repo-url]
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
   - Cases: 13,087
   - Events: 262,200
   
2. **BPI Challenge 2017 (BPI17)**: Credit application process  
   - Cases: 31,509
   - Events: 1,202,267
   
3. **BPI Challenge 2020 - Request for Payment (BPI20RfP)**
   - Cases: 6,886
   - Events: 36,796
   
4. **BPI Challenge 2020 - Prepaid Travel Costs (BPI20PTC)**
   - Cases: 2,099
   - Events: 18,246
   
5. **BPI Challenge 2020 - Permit Data (BPI20PD)**
   - Cases: 7,065  
   - Events: 86,581

### Extension Dataset
6. **BPI Traffic Fines**: Municipal traffic violation process
   - [To be added - dataset characteristics]

## Preprocessing

All datasets undergo consistent preprocessing following the original methodology:

1. **Data Cleaning**: Removal of cases with fewer than two events
2. **Feature Engineering**: Extraction of time-based features from timestamps
3. **Normalization**: Z-score normalization of numerical features
4. **Encoding**: Activity label indexing for categorical embedding layers
5. **Splitting**: Application of unbiased train/test splits to prevent data leakage

Preprocessing is handled automatically when running experiments. Manual preprocessing can be performed using:
```bash
python preprocess_datasets.py --dataset [dataset_name]
```

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

### Quick Start (Jupyter Notebooks)

#### For TU/e HPC Users:
1. **Launch Jupyter** on TU/e HPC with GPU access
2. **Upload notebooks** to your Jupyter session
3. **Run in sequence**:

```bash
# 1. Environment setup and testing
01_setup_and_environment.ipynb

# 2. Execute all experiments  
02_main_experiments.ipynb

# 3. Analyze results and generate reports
03_results_analysis.ipynb
```

### Command Line Execution

#### RNN Baseline Example:
```bash
python next_event_prediction.py \
  --dataset BPI12 \
  --backbone rnn \
  --embedding_size 128 \
  --hidden_size 256 \
  --lr 0.0001 \
  --batch_size 64 \
  --epochs 25 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time
```

#### LLM-PEFT Example (LoRA):
```bash
python next_event_prediction.py \
  --dataset BPI12 \
  --backbone qwen25-05b \
  --embedding_size 896 \
  --hidden_size 896 \
  --lr 0.00005 \
  --batch_size 64 \
  --epochs 10 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time \
  --fine_tuning lora \
  --r 256 \
  --lora_alpha 512
```

#### LLM-PEFT Example (Layer Freezing):
```bash
python next_event_prediction.py \
  --dataset BPI12 \
  --backbone llama32-1b \
  --embedding_size 2048 \
  --hidden_size 2048 \
  --lr 0.00005 \
  --batch_size 64 \
  --epochs 10 \
  --categorical_features activity \
  --continuous_features all \
  --categorical_targets activity \
  --continuous_targets remaining_time \
  --fine_tuning freeze \
  --freeze_layers -1,-2
```

### Competitor Baselines

#### S-NAP (Narrative Approach):
```bash
python rebmann_et_al.py --dataset BPI12 --epochs 10
```

#### Transfer Learning Baseline:
```bash
python luijken_transfer_learning.py --dataset BPI12 --epochs 10
```

## Evaluation

### Metrics
- **Next Activity Prediction**: Classification accuracy
- **Remaining Time Prediction**: Mean Squared Error (MSE)  
- **Training Efficiency**: Runtime and parameter count
- **Convergence Analysis**: Loss curves and training stability

### Evaluation Scripts

#### Evaluate Individual Model:
```bash
python evaluate.py --model_path ./checkpoints/model_name --dataset BPI12
```

#### Ensemble Evaluation:
```bash
python evaluate.py --ensemble_paths ./checkpoints/model1,./checkpoints/model2,./checkpoints/model3 --dataset BPI12
```

### Performance Analysis
Results are automatically analyzed and visualized when using the Jupyter notebooks. Manual analysis can be performed using:
```bash
python analyze_results.py --results_dir ./replication_results
```

## Repository Structure (incomplete)

```
llm-peft-ppm-replication/
├── 📓 notebooks/
│   ├── 01_setup_and_environment.ipynb     # Environment setup
│   ├── 02_main_experiments.ipynb          # Core experiments
│   └── 03_results_analysis.ipynb          # Analysis & visualization
├── 📁 original_repo/                      # Original code (auto-cloned)
├── 📁 replication_results/                # Generated results
│   ├── experiments/                       # Raw outputs
│   ├── logs/                             # Training logs
│   ├── plots/                            # Visualizations
│   └── *.csv                             # Performance summaries
├── 📁 data/                               # Datasets (auto-downloaded)
├── 📄 requirements.txt                    # Dependencies
├── 📄 README.md                          # This file
<!-- └── 📄 methodology.md                     # Detailed methodology -->
```

<!-- ## Research Questions Addressed

This replication study investigates the following research questions from the original paper:

### RQ1: Performance Comparison
**Do PEFT-adapted LLMs outperform traditional RNN-based approaches and narrative-style LLM methods in predictive process monitoring?**

- Statistical comparison between RNN baselines and LLM-PEFT methods
- Performance evaluation across all datasets
- Significance testing and effect size analysis

### RQ2: Multi-task Learning Effectiveness  
**Does multi-task learning (joint prediction of next activity and remaining time) provide more robust and consistent performance compared to single-task approaches?**

- Comparison of single-task vs. multi-task configurations
- Robustness analysis across different datasets
- Training stability assessment

### RQ3: PEFT Strategy Optimization
**What are the optimal PEFT strategies (LoRA vs. layer freezing) for different prediction tasks?**

- Comprehensive comparison of PEFT techniques
- Task-specific strategy recommendations
- Computational efficiency analysis

### Extension: Generalizability Assessment
**How do the proposed approaches generalize to different types of business processes?**

- Performance evaluation on BPI Traffic Fines dataset
- Cross-domain generalization analysis
- Process type characteristics impact study

## Expected Results

Based on the original study, we expect to observe:

1. **Superior LLM-PEFT Performance**: PEFT-adapted LLMs should outperform traditional RNN approaches
2. **Multi-task Advantages**: Joint training should provide more robust performance than single-task approaches  
3. **PEFT Strategy Differences**: LoRA and layer freezing strategies should show varying effectiveness across tasks
4. **Faster Convergence**: LLMs should require minimal hyperparameter optimization and demonstrate faster convergence

Detailed results and statistical analysis will be provided upon completion of the experimental evaluation. -->

## Reproducibility

### Computational Environment
- **Hardware**: TU/e HPC with NVIDIA GPUs
- **Software**: PyTorch 2.5.1, CUDA 11.7+
- **Random Seeds**: Fixed for reproducibility
- **Checkpointing**: Models saved for result verification

### Data Availability  
- All datasets are publicly available through SkPM
- Preprocessing scripts ensure consistent data preparation
- Train/test splits follow original methodology exactly

### Code Availability
- Complete replication code provided in this repository
- Original authors' code integrated and properly attributed
- Jupyter notebooks provide step-by-step execution guide

## Results Summary

[To be completed upon experiment completion]

### Performance Comparison
- RNN Baseline: [Results TBA]
- LLM-PEFT (Best): [Results TBA]  
- Statistical Significance: [Analysis TBA]

### PEFT Strategy Analysis
- LoRA Performance: [Results TBA]
- Layer Freezing Performance: [Results TBA]
- Optimal Strategy: [Conclusion TBA]

### Replication Fidelity
- Reproduction Accuracy: [Assessment TBA]
- Key Findings Alignment: [Analysis TBA]

## Discussion

[To be completed - will include:]
- Comparison with original study results
- Methodological differences and their impact
- Limitations and potential improvements
- Implications for the process mining community

## Citation

<!-- If you use this replication study in your research, please cite:

```bibtex
@misc{llm_peft_ppm_replication_2025,
  title={Replication Study: Parameter-Efficient Fine-Tuning of Large Language Models for Predictive Process Monitoring},
  author={[Your Name]},
  year={2025},
  institution={Eindhoven University of Technology},
  url={[Your Repository URL]}
}
``` -->

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

