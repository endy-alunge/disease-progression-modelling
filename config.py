"""
config.py
---------
Central configuration — all paths, hyperparameters, and constants.
Change values here; they propagate everywhere automatically.
"""

import os
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Set this to your actual CSV path:
CSV_PATH    = os.path.join(DATA_DIR, "bcw_synth.csv")

# ── Column schema ─────────────────────────────────────────────────────────────
ID_COL          = "id"
DIAGNOSIS_COL   = "diagnosis"      # M = malignant, B = benign
YEAR_COL        = "year"

# All 30 feature columns (mean, se, worst)
FEATURE_COLS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# ── Clinical Staging (NEW) ────────────────────────────────────────────────────
# Derive cancer stages I-IV from tumor characteristics
# Stage 0: Benign (no cancer)
# Stage 1: Stage I (small tumor, < 2cm)
# Stage 2: Stage II (2-5cm or lymph node involvement)
# Stage 3: Stage III (larger tumor, >5cm)
# Stage 4: Stage IV (metastatic)

STAGE_CLASSES = ["Benign", "Stage I", "Stage II", "Stage III", "Stage IV"]
N_STAGES = 5

# Staging thresholds based on perimeter (proxy for tumor size)
# Derived from the data distribution
STAGE_THRESHOLDS = {
    "benign_max": 90,      # Perimeter < 90 → Benign
    "stage1_max": 110,     # 90-110 → Stage I
    "stage2_max": 130,     # 110-130 → Stage II
    "stage3_max": 160,     # 130-160 → Stage III
    # > 160 → Stage IV
}

# ── Target definitions ─────────────────────────────────────────────────────────
# Task 1 — predict diagnosis label at each time step (binary)
DIAGNOSIS_CLASSES   = ["B", "M"]          # 0=Benign, 1=Malignant
N_DIAGNOSIS_CLASSES = 2

# Task 2 — predict whether diagnosis changes next step (binary)
PROGRESSION_CLASSES   = ["Stable", "Progressed"]   # 0=no change, 1=changed
N_PROGRESSION_CLASSES = 2

# Task 3 — predict clinical stage (NEW, more granular)
# This is your main task for better results
STAGE_TASK_CLASSES = STAGE_CLASSES
N_STAGE_CLASSES = N_STAGES

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# ── Sequence settings ─────────────────────────────────────────────────────────
MAX_SEQ_LEN = 10          # pad/truncate all sequences to this length
MIN_SEQ_LEN = 3           # minimum sequence length to include (NEW)

# ── Markov Chain (IMPROVED) ───────────────────────────────────────────────────
MC_SMOOTHING = 0.1        # Laplace smoothing (was 1e-6 → too small!)
# Higher smoothing prevents zero probabilities for unseen transitions

# ── HMM (IMPROVED) ───────────────────────────────────────────────────────────
HMM_N_HIDDEN    = 8        # increased from 4 (more expressive)
HMM_N_ITER      = 300      # kept high for convergence
HMM_TOL         = 1e-4     # kept from original
HMM_N_RESTARTS  = 5        # NEW: multiple restarts to avoid local optima

# ── RNN shared (IMPROVED) ─────────────────────────────────────────────────────
RNN_HIDDEN_DIM  = 128      # increased from 64 (more capacity)
RNN_N_LAYERS    = 2
RNN_DROPOUT     = 0.3
RNN_LR          = 1e-3
RNN_WEIGHT_DECAY = 1e-5    # NEW: L2 regularization
RNN_EPOCHS      = 100      # increased from 60
RNN_BATCH_SIZE  = 32
RNN_PATIENCE    = 15       # NEW: early stopping patience

# RNN architecture options
RNN_BIDIRECTIONAL = True   # NEW: use bidirectional layers
RNN_USE_ATTENTION = True   # NEW: use attention mechanism

# ── Training improvements (NEW) ───────────────────────────────────────────────
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0

USE_SCHEDULER = True
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

# ── Feature engineering (NEW) ─────────────────────────────────────────────────
# Whether to add temporal features (time since first visit, etc.)
ADD_TEMPORAL_FEATURES = True

# Whether to normalize features
NORMALIZE_FEATURES = True
NORMALIZATION_METHOD = "standard"  # 'standard' or 'minmax'

# ── Evaluation (NEW) ─────────────────────────────────────────────────────────
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]

# ── Visualization (NEW) ───────────────────────────────────────────────────────
PLOT_TRANSITION_MATRICES = True
PLOT_TRAINING_HISTORY = True
PLOT_HIDDEN_STATES = True

# ── Random seed for reproducibility ───────────────────────────────────────────
RANDOM_STATE = SEED

# ── Ensure output dirs exist ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helper functions (NEW) ────────────────────────────────────────────────────
def get_stage_from_perimeter(perimeter, diagnosis):
    """
    Convert tumor perimeter and diagnosis to clinical stage.
    
    Parameters:
    -----------
    perimeter : float
        perimeter_mean value from dataset
    diagnosis : str
        'M' for malignant, 'B' for benign
    
    Returns:
    --------
    int : stage code (0-4)
    """
    if diagnosis == 'B':
        return 0  # Benign
    
    # Malignant cases
    if perimeter < STAGE_THRESHOLDS["stage1_max"]:
        return 1  # Stage I
    elif perimeter < STAGE_THRESHOLDS["stage2_max"]:
        return 2  # Stage II
    elif perimeter < STAGE_THRESHOLDS["stage3_max"]:
        return 3  # Stage III
    else:
        return 4  # Stage IV


def get_stage_name(stage_code):
    """Convert stage code to readable name."""
    return STAGE_CLASSES[stage_code]


def get_transition_prior(n_states=N_STAGES):
    """
    Create prior transition matrix for HMM initialization.
    Assumes higher probability of staying in same state.
    """
    transmat = np.eye(n_states) * 0.7
    transmat = transmat + np.ones((n_states, n_states)) * 0.3 / n_states
    return transmat / transmat.sum(axis=1, keepdims=True)