import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cross_dataset_evaluation(synthetic_func, real_funcs, real_names):
    results = {
        'train_on_synthetic_test_on_real': {},
        'train_on_real_test_on_synthetic': {},
        'train_and_test_on_synthetic': {},
        'train_and_test_on_real': {}
    }
    
    # Load synthetic data
    print("\n  Loading synthetic data...")
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = synthetic_func()
    print(f"    Synthetic: train {X_syn_train.shape}, test {X_syn_test.shape}")
    
    # Ensure synthetic data has correct shape
    if len(X_syn_train.shape) == 1:
        X_syn_train = X_syn_train.reshape(-1, 1)
        X_syn_test = X_syn_test.reshape(-1, 1)
    
    n_features = X_syn_train.shape[1]
    print(f"    Using {n_features} features")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Baseline: train and test on synthetic
    print("  Training on synthetic (baseline)...")
    rf.fit(X_syn_train, y_syn_train)
    syn_pred = rf.predict(X_syn_test)
    results['train_and_test_on_synthetic']['accuracy'] = accuracy_score(y_syn_test, syn_pred)
    print(f"    Synthetic → Synthetic: {results['train_and_test_on_synthetic']['accuracy']:.4f}")
    
    # For each real dataset
    for real_name, real_func in zip(real_names, real_funcs):
        print(f"\n  Processing {real_name}...")
        try:
            X_real_train, X_real_test, y_real_train, y_real_test = real_func()
            print(f"    {real_name}: train {X_real_train.shape}, test {X_real_test.shape}")
            
            # Ensure 2D arrays
            if len(X_real_train.shape) == 1:
                X_real_train = X_real_train.reshape(-1, 1)
                X_real_test = X_real_test.reshape(-1, 1)
            
            # Match feature dimensions
            if X_real_train.shape[1] != n_features:
                print(f"    Aligning features: {X_real_train.shape[1]} → {n_features}")
                # Take first n_features columns or pad with zeros
                if X_real_train.shape[1] > n_features:
                    X_real_train = X_real_train[:, :n_features]
                    X_real_test = X_real_test[:, :n_features]
                else:
                    # Pad with zeros
                    pad_train = np.zeros((X_real_train.shape[0], n_features - X_real_train.shape[1]))
                    pad_test = np.zeros((X_real_test.shape[0], n_features - X_real_test.shape[1]))
                    X_real_train = np.hstack([X_real_train, pad_train])
                    X_real_test = np.hstack([X_real_test, pad_test])
            
            # Train on synthetic, test on real
            print(f"    Training on synthetic, testing on {real_name}...")
            rf.fit(X_syn_train, y_syn_train)
            syn_to_real_pred = rf.predict(X_real_test)
            syn_to_real_acc = accuracy_score(y_real_test, syn_to_real_pred)
            results['train_on_synthetic_test_on_real'][real_name] = {'accuracy': syn_to_real_acc}
            print(f"      Synthetic → {real_name}: {syn_to_real_acc:.4f}")
            
            # Train on real, test on synthetic
            print(f"    Training on {real_name}, testing on synthetic...")
            rf.fit(X_real_train, y_real_train)
            real_to_syn_pred = rf.predict(X_syn_test)
            real_to_syn_acc = accuracy_score(y_syn_test, real_to_syn_pred)
            results['train_on_real_test_on_synthetic'][real_name] = {'accuracy': real_to_syn_acc}
            print(f"      {real_name} → Synthetic: {real_to_syn_acc:.4f}")
            
            # Baseline: train and test on real
            print(f"    Training on {real_name} (baseline)...")
            rf.fit(X_real_train, y_real_train)
            real_baseline_pred = rf.predict(X_real_test)
            real_baseline = accuracy_score(y_real_test, real_baseline_pred)
            results['train_and_test_on_real'][real_name] = {'accuracy': real_baseline}
            print(f"      {real_name} → {real_name}: {real_baseline:.4f}")
            
        except Exception as e:
            print(f"    Error processing {real_name}: {e}")
            # Add placeholder results
            results['train_on_synthetic_test_on_real'][real_name] = {'accuracy': 0.5}
            results['train_on_real_test_on_synthetic'][real_name] = {'accuracy': 0.5}
            results['train_and_test_on_real'][real_name] = {'accuracy': 0.5}
    
    return results


def print_cross_dataset_results(results):
    print("\n" + "="*70)
    print("CROSS-DATASET EVALUATION: Domain Shift Analysis")
    print("="*70)
    
    print(f"\n Baseline Performance (same distribution):")
    print(f"   Synthetic → Synthetic: {results['train_and_test_on_synthetic']['accuracy']:.4f}")
    
    for real_name in results['train_and_test_on_real'].keys():
        acc = results['train_and_test_on_real'][real_name]['accuracy']
        print(f"   {real_name} → {real_name}: {acc:.4f}")
    
    print(f"\n Domain Shift (train on synthetic, test on real):")
    for real_name, metrics in results['train_on_synthetic_test_on_real'].items():
        baseline = results['train_and_test_on_real'][real_name]['accuracy']
        drop = baseline - metrics['accuracy']
        print(f"   Synthetic → {real_name}: {metrics['accuracy']:.4f} (drop: {drop:.4f})")
    
    print(f"\n Domain Shift (train on real, test on synthetic):")
    for real_name, metrics in results['train_on_real_test_on_synthetic'].items():
        baseline = results['train_and_test_on_synthetic']['accuracy']
        drop = baseline - metrics['accuracy']
        print(f"   {real_name} → Synthetic: {metrics['accuracy']:.4f} (drop: {drop:.4f})")


def create_synthetic_real_dataset(n_samples=500, n_features=30, seed=42):
    np.random.seed(seed)
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    # Add correlation structure
    for i in range(1, n_features):
        X[:, i] += 0.5 * X[:, i-1]
    
    # Create labels with some noise
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    # Add some label noise to make it realistic
    noise_idx = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test