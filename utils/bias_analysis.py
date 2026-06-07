import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def run_bias_analysis(synthetic_func, synthetic_name="Synthetic", save_dir=None):
    from utils.real_datasets import load_wdbc, load_wpbc
    
    results = {}
    
    # 1. Load synthetic data
    print("\n  Loading synthetic data (first time point only)...")
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = synthetic_func()
    
    # 2. Load real datasets
    print("\n  Loading real datasets...")
    X_wdbc_train, X_wdbc_test, y_wdbc_train, y_wdbc_test = load_wdbc()
    X_wpbc_train, X_wpbc_test, y_wpbc_train, y_wpbc_test = load_wpbc()
    
    # 3. Train Random Forest on each
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("\n  Training on synthetic data...")
    rf.fit(X_syn_train, y_syn_train)
    y_syn_pred = rf.predict(X_syn_test)
    syn_acc = accuracy_score(y_syn_test, y_syn_pred)
    results['synthetic'] = {
        'accuracy': syn_acc, 
        'predictions': y_syn_pred, 
        'true': y_syn_test,
        'name': synthetic_name
    }
    print(f"    {synthetic_name} accuracy: {syn_acc:.4f}")
    
    print("  Training on WDBC (real diagnostic)...")
    rf.fit(X_wdbc_train, y_wdbc_train)
    y_wdbc_pred = rf.predict(X_wdbc_test)
    wdbc_acc = accuracy_score(y_wdbc_test, y_wdbc_pred)
    results['wdbc'] = {
        'accuracy': wdbc_acc, 
        'predictions': y_wdbc_pred, 
        'true': y_wdbc_test,
        'name': 'WDBC'
    }
    print(f"    WDBC accuracy: {wdbc_acc:.4f}")
    
    print("  Training on WPBC (real prognostic)...")
    rf.fit(X_wpbc_train, y_wpbc_train)
    y_wpbc_pred = rf.predict(X_wpbc_test)
    wpbc_acc = accuracy_score(y_wpbc_test, y_wpbc_pred)
    results['wpbc'] = {
        'accuracy': wpbc_acc, 
        'predictions': y_wpbc_pred, 
        'true': y_wpbc_test,
        'name': 'WPBC'
    }
    print(f"    WPBC accuracy: {wpbc_acc:.4f}")
    
    # 4. Print bias summary
    print("\n" + "="*60)
    print("BIAS ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<25} {'Accuracy':<12} {'Difference':<15}")
    print("-" * 55)
    print(f"{synthetic_name:<25} {syn_acc:.4f} {'(baseline)':<15}")
    print(f"{'WDBC (real diagnostic)':<25} {wdbc_acc:.4f} {(syn_acc - wdbc_acc):+.4f}")
    print(f"{'WPBC (real prognostic)':<25} {wpbc_acc:.4f} {(syn_acc - wpbc_acc):+.4f}")
    
    
    print("\nKEY INSIGHT:")
    print(f"   Synthetic data overestimates performance by {((syn_acc - wdbc_acc) * 100):.1f}%")
    print(f"   compared to real diagnostic data.")
    print(f"\n   → Models trained on synthetic data need recalibration")
    print(f"     before clinical deployment on real patient data.")
    
    # 5. Plot comparison
    if save_dir:
        _plot_bias_comparison(results, save_dir)
        _plot_confusion_comparison(results, save_dir)
    
    return results


def compute_bias_metrics(y_synthetic, y_real, dataset_names):
    metrics = {}
    y_synth_flat = np.array(y_synthetic).flatten()
    
    for real_name, y_real_pred in zip(dataset_names, y_real):
        y_real_flat = np.array(y_real_pred).flatten()
        
        metrics[real_name] = {
            'mean_shift': np.mean(y_synth_flat) - np.mean(y_real_flat),
            'std_ratio': np.std(y_synth_flat) / (np.std(y_real_flat) + 1e-8),
            'agreement_rate': np.mean(y_synth_flat == y_real_flat)
        }
    
    return metrics


def _plot_bias_comparison(results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [results[d]['name'] for d in ['synthetic', 'wdbc', 'wpbc']]
    accuracies = [results[d]['accuracy'] for d in ['synthetic', 'wdbc', 'wpbc']]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(datasets, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Performance: Synthetic vs Real Breast Cancer Data', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Clinical Acceptability')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bias_accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Bias plot saved to {save_dir}/bias_accuracy_comparison.png")


def _plot_confusion_comparison(results, save_dir):
    datasets = list(results.keys())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        cm = confusion_matrix(results[dataset]['true'], results[dataset]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        axes[idx].set_title(f"{results[dataset]['name']}\nAcc: {results[dataset]['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.suptitle('Confusion Matrices: Synthetic vs Real Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/bias_confusion_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion comparison saved to {save_dir}/bias_confusion_comparison.png")