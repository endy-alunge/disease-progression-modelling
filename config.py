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

# ── Clinical Staging ─────────────────────────────────────────────────────────
STAGE_CLASSES = ["Benign", "Stage I", "Stage II", "Stage III", "Stage IV"]
N_STAGES = 5

# Staging thresholds based on perimeter
STAGE_THRESHOLDS = {
    "benign_max": 90,   # Perimeter < 90 → Benign
    "stage1_max": 110,  # 90-110 → Stage I
    "stage2_max": 130,  # 110-130 → Stage II
    "stage3_max": 160,  # 130-160 → Stage III
    # > 160 → Stage IV
}

# ── Target definitions ─────────────────────────────────────────────────────────
DIAGNOSIS_CLASSES   = ["B", "M"]
N_DIAGNOSIS_CLASSES = 2

PROGRESSION_CLASSES   = ["Stable", "Progressed"]
N_PROGRESSION_CLASSES = 2

# ── Dataset split ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

# ── Sequence settings ─────────────────────────────────────────────────────────
MAX_SEQ_LEN = 10
MIN_SEQ_LEN = 2

# ── Feature engineering ───────────────────────────────────────────────────────
NORMALIZE_FEATURES = True
NORMALIZATION_METHOD = "standard"  # 'standard' or 'minmax'
ADD_TEMPORAL_FEATURES = False  # Set to False to avoid dimension mismatch

# ── Markov Chain ──────────────────────────────────────────────────────────────
MC_SMOOTHING = 0.1

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_HIDDEN    = 8
HMM_N_ITER      = 300
HMM_TOL         = 1e-4
HMM_N_RESTARTS  = 5

# ── RNN shared ────────────────────────────────────────────────────────────────
RNN_HIDDEN_DIM  = 128
RNN_N_LAYERS    = 2
RNN_DROPOUT     = 0.3
RNN_LR          = 1e-3
RNN_WEIGHT_DECAY = 1e-5
RNN_EPOCHS      = 100
RNN_BATCH_SIZE  = 32
RNN_PATIENCE    = 10
RNN_BIDIRECTIONAL = True

# ── Training improvements ─────────────────────────────────────────────────────
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_STATE = SEED

# ── Ensure output dirs exist ──────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def get_stage_from_perimeter(perimeter, diagnosis):
    if diagnosis == 'B':
        return 0  # Benign
    
    if perimeter < STAGE_THRESHOLDS["stage1_max"]:
        return 1  # Stage I
    elif perimeter < STAGE_THRESHOLDS["stage2_max"]:
        return 2  # Stage II
    elif perimeter < STAGE_THRESHOLDS["stage3_max"]:
        return 3  # Stage III
    else:
        return 4  # Stage IV


def get_stage_name(stage_code):
    return STAGE_CLASSES[stage_code]


def get_transition_prior(n_states=N_STAGES):
    transmat = np.eye(n_states) * 0.7
    transmat = transmat + np.ones((n_states, n_states)) * 0.3 / n_states
    return transmat / transmat.sum(axis=1, keepdims=True)