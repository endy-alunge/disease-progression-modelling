import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_wdbc():
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=15)
        X = data.data.features
        y = data.data.targets.values.ravel()
    except:
        return _create_fallback_dataset("WDBC", 569, 30)
    
    # Process features
    if isinstance(X, pd.DataFrame):
        # Take first 30 columns
        X = X.iloc[:, :30].values
    else:
        X = np.array(X)[:, :30]
    
    # Handle non-numeric
    X = np.array([[float(x) if x != '?' else np.nan for x in row] for row in X])
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Convert labels
    y = np.array([1 if val == 'M' else 0 for val in y])
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  WDBC: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    return X_train, X_test, y_train, y_test


def _create_fallback_dataset(name, n_samples, n_features):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # Create realistic correlation
    for i in range(1, n_features):
        X[:, i] += 0.3 * X[:, i-1]
    
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Add some imbalance (like real cancer data)
    if y.sum() > n_samples // 2:
        y = 1 - y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  {name} (fallback): {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


def _load_wdbc_fallback():
    np.random.seed(42)
    n_samples = 569
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Ensure some class balance
    if y.sum() < 100:
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  WDBC (fallback): {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    return X_train, X_test, y_train, y_test


def load_wpbc():
    try:
        from ucimlrepo import fetch_ucirepo
        data = fetch_ucirepo(id=16)
        X = data.data.features
        y = data.data.targets.values.ravel()
    except Exception as e:
        print(f"  Error loading WPBC: {e}")
        return _load_wpbc_fallback()
    
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Replace '?' with NaN and convert to numeric
    X = X.replace('?', np.nan)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # Take first 30 columns (or fewer if not available)
    X = X.iloc[:, :min(30, X.shape[1])]
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_array = imputer.fit_transform(X.values)
    
    # Convert R/N to 1/0 (R=recurrence=1)
    y = np.array([1 if val == 'R' else 0 for val in y])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  WPBC: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    print(f"    Recurrence: {y.sum()}, No Recurrence: {len(y) - y.sum()}")
    
    return X_train, X_test, y_train, y_test


def _load_wpbc_fallback():
    np.random.seed(42)
    n_samples = 198
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  WPBC (fallback): {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    return X_train, X_test, y_train, y_test


def compare_datasets_summary():
    print("\n" + "="*60)
    print("DATASET COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<20} {'Type':<15} {'Samples':<10} {'Features':<10} {'Temporal':<10}")
    print("-" * 65)
    print(f"{'Synthetic (WDBC-SL)':<20} {'Longitudinal':<15} {'300+':<10} {'30':<10} {'Yes':<10}")
    print(f"{'WDBC (Diagnostic)':<20} {'Cross-sectional':<15} {'569':<10} {'30':<10} {'No':<10}")
    print(f"{'WPBC (Prognostic)':<20} {'Cross-sectional':<15} {'198':<10} {'30':<10} {'No':<10}")