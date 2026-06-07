import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CSV_PATH, DATA_DIR, FEATURE_COLS, ID_COL, DIAGNOSIS_COL, YEAR_COL, SEED
)


# ── Feature distributions learned from Wisconsin dataset statistics ────────────
# (mean, std) for B and M classes
FEATURE_STATS = {
    # feature: (B_mean, B_std, M_mean, M_std)
    "radius_mean":              (12.15, 1.78, 17.46, 3.20),
    "texture_mean":             (17.91, 4.00, 21.60, 3.78),
    "perimeter_mean":           (78.08, 11.84, 115.37, 21.89),
    "area_mean":                (462.79, 134.29, 978.38, 367.55),
    "smoothness_mean":          (0.092, 0.013, 0.103, 0.013),
    "compactness_mean":         (0.080, 0.034, 0.145, 0.053),
    "concavity_mean":           (0.046, 0.043, 0.161, 0.097),
    "concave points_mean":      (0.026, 0.016, 0.088, 0.036),
    "symmetry_mean":            (0.174, 0.019, 0.193, 0.027),
    "fractal_dimension_mean":   (0.063, 0.007, 0.062, 0.008),
    "radius_se":                (0.284, 0.113, 0.609, 0.345),
    "texture_se":               (1.220, 0.551, 1.210, 0.486),
    "perimeter_se":             (2.000, 0.828, 4.323, 2.438),
    "area_se":                  (21.14, 11.00, 72.67, 58.44),
    "smoothness_se":            (0.007, 0.002, 0.007, 0.003),
    "compactness_se":           (0.021, 0.013, 0.032, 0.019),
    "concavity_se":             (0.026, 0.026, 0.042, 0.036),
    "concave points_se":        (0.010, 0.006, 0.015, 0.008),
    "symmetry_se":              (0.021, 0.008, 0.021, 0.008),
    "fractal_dimension_se":     (0.004, 0.001, 0.004, 0.002),
    "radius_worst":             (13.38, 2.00, 21.13, 4.29),
    "texture_worst":            (23.52, 6.00, 29.32, 6.15),
    "perimeter_worst":          (87.01, 13.45, 141.37, 29.89),
    "area_worst":               (558.90, 161.40, 1422.29, 589.71),
    "smoothness_worst":         (0.125, 0.019, 0.145, 0.022),
    "compactness_worst":        (0.178, 0.079, 0.374, 0.162),
    "concavity_worst":          (0.166, 0.127, 0.451, 0.200),
    "concave points_worst":     (0.074, 0.030, 0.182, 0.059),
    "symmetry_worst":           (0.271, 0.046, 0.323, 0.062),
    "fractal_dimension_worst":  (0.079, 0.014, 0.092, 0.018),
}

# Transition probs: P(next_diag | current_diag)
DIAG_TRANSITION = {
    "B": [0.90, 0.10],   # benign mostly stays benign
    "M": [0.05, 0.95],   # malignant rarely remits
}


def _patient_id(n=5):
    chars = "0123456789abcdef"
    return "".join(np.random.choice(list(chars), n))


def generate_patient(pid, n_years, start_diag, rng):
    rows = []
    diag = start_diag
    for yr in range(n_years):
        row = {ID_COL: pid, DIAGNOSIS_COL: diag, YEAR_COL: yr}
        for feat, (bm, bs, mm, ms) in FEATURE_STATS.items():
            mu, sigma = (bm, bs) if diag == "B" else (mm, ms)
            # Add small longitudinal drift (noise across years)
            val = rng.normal(mu + rng.normal(0, sigma * 0.05), sigma * 0.15)
            row[feat] = round(max(val, 0.0001), 4)
        rows.append(row)
        # Transition to next year's diagnosis
        probs = DIAG_TRANSITION[diag]
        diag = rng.choice(["B", "M"], p=probs)
    return rows


def generate_dataset(n_patients=300, min_years=4, max_years=8, output_path=None, seed=SEED):
    rng = np.random.default_rng(seed)
    all_rows = []
    seen_ids = set()

    for _ in range(n_patients):
        # Unique 5-char hex id
        pid = _patient_id()
        while pid in seen_ids:
            pid = _patient_id()
        seen_ids.add(pid)

        n_years    = int(rng.integers(min_years, max_years + 1))
        start_diag = rng.choice(["B", "M"], p=[0.63, 0.37])  # ~37% malignant
        all_rows.extend(generate_patient(pid, n_years, start_diag, rng))

    df = pd.DataFrame(all_rows, columns=[ID_COL, DIAGNOSIS_COL, YEAR_COL] + FEATURE_COLS)
    df = df.sort_values([ID_COL, YEAR_COL]).reset_index(drop=True)

    save_path = output_path or CSV_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Synthetic dataset saved → {save_path}")
    print(f"  Patients: {n_patients} | Total rows: {len(df)}")
    return df


if __name__ == "__main__":
    df = generate_dataset(300)
    print(df.head(12).to_string(index=False))