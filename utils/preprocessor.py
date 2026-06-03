"""
utils/preprocessor_improved.py
-------------------------------
Improved preprocessor with clinical stage support and better RNN data preparation.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CSV_PATH, ID_COL, DIAGNOSIS_COL, YEAR_COL, FEATURE_COLS,
    DIAGNOSIS_CLASSES, MAX_SEQ_LEN, MIN_SEQ_LEN,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
    STAGE_THRESHOLDS, STAGE_CLASSES, N_STAGES,
    NORMALIZE_FEATURES, ADD_TEMPORAL_FEATURES,
)

DIAG_TO_INT = {"B": 0, "M": 1}
INT_TO_DIAG = {0: "B", 1: "M"}


# ── Clinical Stage Functions (NEW) ────────────────────────────────────────────
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


def add_stage_labels(df):
    """Add clinical stage column based on perimeter and diagnosis"""
    df['stage'] = df.apply(
        lambda row: get_stage_from_perimeter(row['perimeter_mean'], row['diagnosis']),
        axis=1
    )
    return df


# ── CSV loader (updated with stage labels) ────────────────────────────────────
def load_csv(path=CSV_PATH):
    """Load CSV, add stage labels, sort by patient + year, return DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found at: {path}\n"
            "Either place your file there or run:\n"
            "  python utils/data_generator.py"
        )
    df = pd.read_csv(path)
    df[YEAR_COL]      = df[YEAR_COL].astype(int)
    df[DIAGNOSIS_COL] = df[DIAGNOSIS_COL].str.strip().str.upper()
    
    # Add clinical stages
    df = add_stage_labels(df)
    
    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {df[ID_COL].nunique()} patients from {path}")
    
    # Print stage distribution
    stage_counts = df['stage'].value_counts().sort_index()
    print(f"  Stage distribution: {dict(zip(STAGE_CLASSES, stage_counts.values))}")
    
    return df


# ── Build patient-level sequences (UPDATED with stages) ───────────────────────
def build_patient_sequences(df):
    """
    Returns a list of dicts, one per patient:
      {
        'id':        patient id,
        'diag_seq':  [0,0,1,1,...]   int label per year,
        'prog_seq':  [0,1,0,...]     did label CHANGE at next step,
        'stage_seq': [0,1,2,3,4,...] clinical stage per year (NEW),
        'feat_seq':  np.array shape (T, 30),
        'years':     [0,1,2,...],
      }
    """
    patients = []
    for pid, grp in df.groupby(ID_COL):
        grp = grp.sort_values(YEAR_COL)
        diag_seq = [DIAG_TO_INT[d] for d in grp[DIAGNOSIS_COL]]
        stage_seq = grp['stage'].tolist()  # NEW: clinical stages
        feat_seq = grp[FEATURE_COLS].values.astype(np.float32)
        years    = grp[YEAR_COL].tolist()

        # Progression: 1 if label changes at the NEXT step
        prog_seq = [
            int(diag_seq[t] != diag_seq[t + 1])
            for t in range(len(diag_seq) - 1)
        ]
        
        # Stage progression: 1 if stage increases (NEW)
        stage_prog_seq = [
            int(stage_seq[t + 1] > stage_seq[t])
            for t in range(len(stage_seq) - 1)
        ]

        if len(diag_seq) < MIN_SEQ_LEN:
            continue   # need at least MIN_SEQ_LEN time points

        patients.append({
            "id":              pid,
            "diag_seq":        diag_seq,
            "prog_seq":        prog_seq,
            "stage_seq":       stage_seq,          # NEW
            "stage_prog_seq":  stage_prog_seq,     # NEW
            "feat_seq":        feat_seq,
            "years":           years,
        })
    return patients


# ── Train / val / test split (patient-level, stratified by first stage) ───────
def split_patients(patients, seed=SEED):
    """Split patients with stratification by first clinical stage"""
    n = len(patients)
    
    # Get first stage for stratification
    first_stages = [p["stage_seq"][0] for p in patients]
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        range(n), 
        test_size=TEST_RATIO, 
        stratify=first_stages,
        random_state=seed
    )
    
    # Second split: train vs val
    train_val_stages = [first_stages[i] for i in train_val_idx]
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=train_val_stages,
        random_state=seed
    )
    
    return (
        [patients[i] for i in train_idx],
        [patients[i] for i in val_idx],
        [patients[i] for i in test_idx],
    )


# ── Normalise features (improved) ─────────────────────────────────────────────
def fit_scaler(train_patients, method="standard"):
    """Fit StandardScaler or MinMaxScaler on training feature rows."""
    all_feats = np.vstack([p["feat_seq"] for p in train_patients])
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    scaler.fit(all_feats)
    return scaler


def apply_scaler(patients, scaler):
    """Apply fitted scaler to a list of patients (in-place copy)."""
    scaled = []
    for p in patients:
        pc = dict(p)
        pc["feat_seq"] = scaler.transform(p["feat_seq"]).astype(np.float32)
        scaled.append(pc)
    return scaled


def add_temporal_features(patients):
    """Add time since first visit as additional feature (NEW)"""
    for p in patients:
        years = np.array(p["years"]).reshape(-1, 1)
        time_since_first = years - years[0]
        # Normalize by max time
        if len(time_since_first) > 1:
            time_since_first = time_since_first / (time_since_first[-1] + 1e-8)
        
        # Append to feature sequence
        feat_seq = p["feat_seq"]
        p["feat_seq"] = np.hstack([feat_seq, time_since_first])
    
    return patients


# ── Markov / HMM format (UPDATED with stages) ─────────────────────────────────
def patients_to_diag_sequences(patients):
    """Extract plain integer diagnosis sequences for Markov Chain / HMM."""
    return [p["diag_seq"] for p in patients]


def patients_to_stage_sequences(patients):
    """Extract plain integer stage sequences for Markov Chain / HMM (NEW)."""
    return [p["stage_seq"] for p in patients]


def patients_to_prog_sequences(patients):
    """Extract progression sequences."""
    return [p["prog_seq"] for p in patients]


def patients_to_stage_prog_sequences(patients):
    """Extract stage progression sequences (NEW)."""
    return [p["stage_prog_seq"] for p in patients]


# ── RNN / supervised format (IMPROVED with better padding) ────────────────────
def build_rnn_dataset(patients, task="diag", max_len=MAX_SEQ_LEN, 
                      add_temporal=ADD_TEMPORAL_FEATURES):
    """
    Build padded (X, y) arrays for RNN training with improved handling.

    task options:
      'diag'   → predict diagnosis label at each step
      'prog'   → predict progression at each step
      'stage'  → predict clinical stage at each step (NEW - BEST!)
      'stage_prog' → predict stage progression

    Returns:
      X:      float32  (N, max_len, n_features)
      y:      int64    (N,)
      masks:  bool     (N, max_len) — True = real data
    """
    n_feat = len(FEATURE_COLS)
    if add_temporal:
        n_feat += 1  # add time feature
    
    X_list, y_list, mask_list = [], [], []

    for p in patients:
        T        = len(p["diag_seq"])
        feat_seq = p["feat_seq"]        # (T, n_features)
        
        if task == "diag":
            diag_seq = p["diag_seq"]
            for t in range(1, T):
                history = feat_seq[:t]
                label   = diag_seq[t]
                X_list.append(_pad_seq(history, max_len, n_feat))
                mask_list.append(_build_mask(t, max_len))
                y_list.append(label)
        
        elif task == "prog":
            prog_seq = p["prog_seq"]
            for t in range(len(prog_seq)):
                history = feat_seq[:t + 1]
                label   = prog_seq[t]
                X_list.append(_pad_seq(history, max_len, n_feat))
                mask_list.append(_build_mask(t + 1, max_len))
                y_list.append(label)
        
        elif task == "stage":  # NEW - main task for better results
            stage_seq = p["stage_seq"]
            for t in range(1, T):
                history = feat_seq[:t]
                label   = stage_seq[t]
                X_list.append(_pad_seq(history, max_len, n_feat))
                mask_list.append(_build_mask(t, max_len))
                y_list.append(label)
        
        elif task == "stage_prog":  # NEW
            stage_prog_seq = p["stage_prog_seq"]
            for t in range(len(stage_prog_seq)):
                history = feat_seq[:t + 1]
                label   = stage_prog_seq[t]
                X_list.append(_pad_seq(history, max_len, n_feat))
                mask_list.append(_build_mask(t + 1, max_len))
                y_list.append(label)

    X     = np.stack(X_list).astype(np.float32)
    y     = np.array(y_list, dtype=np.int64)
    masks = np.stack(mask_list)
    return X, y, masks


def _pad_seq(seq, max_len, n_feat):
    """Left-zero-pad a sequence to (max_len, n_feat)."""
    T = len(seq)
    if T >= max_len:
        # Take last max_len timesteps
        return seq[-max_len:]
    pad = np.zeros((max_len - T, n_feat), dtype=np.float32)
    return np.vstack([pad, seq])


def _build_mask(real_len, max_len):
    """Boolean mask: True for real timesteps (right-aligned)."""
    mask = np.zeros(max_len, dtype=bool)
    start = max(0, max_len - real_len)
    mask[start:] = True
    return mask


# ── One-shot pipeline (UPDATED) ───────────────────────────────────────────────
def load_and_prepare(csv_path=CSV_PATH, normalize=True, add_temporal=ADD_TEMPORAL_FEATURES):
    """
    Full pipeline:
      load CSV → add stages → build sequences → split → normalise → build RNN arrays.

    Returns a dict with everything needed by the models.
    """
    df       = load_csv(csv_path)
    patients = build_patient_sequences(df)

    print(f"  Valid patients (≥{MIN_SEQ_LEN} time points): {len(patients)}")
    
    # Statistics
    n_m = sum(1 for p in patients if p["diag_seq"][0] == 1)
    n_b = len(patients) - n_m
    print(f"  Starting: M={n_m} patients | B={n_b} patients")
    
    # Stage distribution at first visit
    first_stages = [p["stage_seq"][0] for p in patients]
    stage_counts = np.bincount(first_stages, minlength=N_STAGES)
    print(f"  First visit stages: {dict(zip(STAGE_CLASSES, stage_counts))}")

    train_p, val_p, test_p = split_patients(patients)
    print(f"  Split → train:{len(train_p)} val:{len(val_p)} test:{len(test_p)}")

    # Normalize features
    if normalize:
        scaler  = fit_scaler(train_p, method="standard")
        train_p = apply_scaler(train_p, scaler)
        val_p   = apply_scaler(val_p,   scaler)
        test_p  = apply_scaler(test_p,  scaler)
    else:
        scaler = None

    # Add temporal features if requested
    if add_temporal:
        train_p = add_temporal_features(train_p)
        val_p = add_temporal_features(val_p)
        test_p = add_temporal_features(test_p)

    # Markov / HMM sequences
    diag_seqs = {
        "train": patients_to_diag_sequences(train_p),
        "val":   patients_to_diag_sequences(val_p),
        "test":  patients_to_diag_sequences(test_p),
    }
    
    # NEW: Stage sequences for better modeling
    stage_seqs = {
        "train": patients_to_stage_sequences(train_p),
        "val":   patients_to_stage_sequences(val_p),
        "test":  patients_to_stage_sequences(test_p),
    }

    # RNN arrays — task 1: diagnosis (binary)
    X_tr_d, y_tr_d, _ = build_rnn_dataset(train_p, task="diag", add_temporal=add_temporal)
    X_va_d, y_va_d, _ = build_rnn_dataset(val_p,   task="diag", add_temporal=add_temporal)
    X_te_d, y_te_d, _ = build_rnn_dataset(test_p,  task="diag", add_temporal=add_temporal)

    # RNN arrays — task 2: progression (binary)
    X_tr_p, y_tr_p, _ = build_rnn_dataset(train_p, task="prog", add_temporal=add_temporal)
    X_va_p, y_va_p, _ = build_rnn_dataset(val_p,   task="prog", add_temporal=add_temporal)
    X_te_p, y_te_p, _ = build_rnn_dataset(test_p,  task="prog", add_temporal=add_temporal)

    # NEW: RNN arrays — task 3: stage prediction (5 classes - BEST!)
    X_tr_s, y_tr_s, _ = build_rnn_dataset(train_p, task="stage", add_temporal=add_temporal)
    X_va_s, y_va_s, _ = build_rnn_dataset(val_p,   task="stage", add_temporal=add_temporal)
    X_te_s, y_te_s, _ = build_rnn_dataset(test_p,  task="stage", add_temporal=add_temporal)

    print(f"\n  RNN diag   — train:{len(y_tr_d)} val:{len(y_va_d)} test:{len(y_te_d)}")
    print(f"  RNN prog   — train:{len(y_tr_p)} val:{len(y_va_p)} test:{len(y_te_p)}")
    print(f"  RNN stage  — train:{len(y_tr_s)} val:{len(y_va_s)} test:{len(y_te_s)}")

    return {
        "scaler":      scaler,
        "patients":    {"train": train_p, "val": val_p, "test": test_p},
        "diag_seqs":   diag_seqs,
        "stage_seqs":  stage_seqs,  # NEW
        "rnn_diag":    {"X_train": X_tr_d, "y_train": y_tr_d,
                        "X_val":   X_va_d, "y_val":   y_va_d,
                        "X_test":  X_te_d, "y_test":  y_te_d},
        "rnn_prog":    {"X_train": X_tr_p, "y_train": y_tr_p,
                        "X_val":   X_va_p, "y_val":   y_va_p,
                        "X_test":  X_te_p, "y_test":  y_te_p},
        "rnn_stage":   {"X_train": X_tr_s, "y_train": y_tr_s,  # NEW - best for RNN
                        "X_val":   X_va_s, "y_val":   y_va_s,
                        "X_test":  X_te_s, "y_test":  y_te_s},
    }


if __name__ == "__main__":
    data = load_and_prepare()
    print("\n" + "="*50)
    print("Sample data:")
    print(f"  First test patient: {data['patients']['test'][0]['id']}")
    print(f"  Diagnosis sequence: {data['patients']['test'][0]['diag_seq']}")
    print(f"  Stage sequence: {data['patients']['test'][0]['stage_seq']}")
    print(f"  Stage names: {[STAGE_CLASSES[s] for s in data['patients']['test'][0]['stage_seq']]}")
    print(f"  RNN stage X shape: {data['rnn_stage']['X_train'].shape}")
    print(f"  RNN stage y sample: {data['rnn_stage']['y_train'][:10]}")