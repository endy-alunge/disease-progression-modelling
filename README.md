# Breast Cancer Progression - Sequence Modelling Pipeline

A comprehensive machine learning framework for modelling disease progression in breast cancer using both probabilistic and deep learning sequence models.

This project compares **Markov Chains**, **Hidden Markov Models (HMM)**, **LSTMs**, and **GRUs** for temporal prediction of clinical states in a synthetic longitudinal breast cancer dataset.

---

# Project Structure
```
breast_cancer_progression/
│
├── main.py # Main pipeline entry point
├── config.py # Hyperparameters & paths
├── requirements.txt # Dependencies
│
├── data/
│ └── bcw_synth.csv # Synthetic longitudinal dataset
│
├── models/
│ ├── markov_chain.py # First-order Markov Chain
│ ├── hmm_model.py # HMM (Baum-Welch + Viterbi)
│ └── rnn_model.py # LSTM & GRU with attention
│
├── utils/
│ ├── preprocessor.py # Sequence builder (diag/stage/feat/prog)
│ ├── evaluation.py # Metrics & visualizations
│ ├── structure_analysis.py # Entropy & determinism analysis
│ ├── real_datasets.py # UCI dataset loaders
│ ├── bias_analysis.py # Synthetic vs real comparison
│ └── cross_dataset_evaluation.py # Domain shift analysis
│
└── outputs/ # Generated figures & results
├── figures/
└── results_summary.txt

## Tasks

Three clinical prediction tasks are evaluated:

Task	Output	Classes
Diagnosis classification	Binary	Benign (0) / Malignant (1)
Progression detection	Binary	Stable (0) / Progressed (1)
Stage classification	Multi-class	Benign, Stage I, II, III, IV
## Dataset

WDBC-SL — Synthetic Longitudinal Breast Cancer Diagnostic Dataset

30 numeric tumour features (radius, texture, perimeter, etc.)
Patient sequences of 5–7 yearly time steps
Diagnosis: Benign (B) / Malignant (M)
Clinical stages derived via rule-based mapping

⚠️ This dataset is synthetic and not clinically deployable.

## Requirements

Install dependencies using:

pip install -r requirements.txt

Or manually:

numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
matplotlib>=3.5.0
seaborn>=0.11.0
hmmlearn>=0.2.8
scipy>=1.7.0
ucimlrepo>=0.0.3

## Usage
# Run full pipeline
python main.py

# Generate synthetic dataset
python main.py --synthetic


## Outputs

All results are saved in the outputs/ folder:

Confusion matrices (diagnosis, progression, stage)
Markov transition heatmaps
HMM structure diagrams
LSTM/GRU training curves
Model comparison charts
Robustness analysis plots
Bias analysis (synthetic vs real)
⚠️ Key Insight

Synthetic data leads to inflated performance estimates (~10% higher than real-world datasets)
