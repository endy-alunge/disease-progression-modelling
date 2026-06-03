"""
utils/evaluation.py
-------------------
Evaluation utilities for sequence models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, model_name, task):
    """Compute standard classification metrics."""
    return {
        "model": model_name,
        "task": task,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def print_report(y_true, y_pred, model_name, task, log_likelihood=None):
    """Print formatted evaluation report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  {model_name} — {task.upper()}")
    print(f"  Accuracy: {acc:.4f}")
    if log_likelihood is not None:
        print(f"  Log-Likelihood: {log_likelihood:.2f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion_matrices(results, task, save_path=None):
    """
    Plot confusion matrices for all models on a given task.
    
    Parameters:
    -----------
    results : list of tuples (model_name, y_true, y_pred)
    task : str ('diag' or 'prog')
    save_path : str, optional
    """
    n_models = len(results)
    if n_models == 0:
        print(f"  No results for task {task}")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, y_true, y_pred) in enumerate(results):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=np.unique(np.concatenate([y_true, y_pred])),
                    yticklabels=np.unique(np.concatenate([y_true, y_pred])))
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.suptitle(f'Confusion Matrices - {task.upper()}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrices saved to {save_path}")
    else:
        plt.show()


def plot_markov_transitions(mc_diag, mc_prog, save_path=None):
    """Plot Markov Chain transition matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Diagnosis transitions
    sns.heatmap(mc_diag.P, annot=True, fmt='.3f', cmap='Blues', ax=axes[0],
                xticklabels=mc_diag.class_names, yticklabels=mc_diag.class_names)
    axes[0].set_title('Diagnosis Transitions')
    axes[0].set_xlabel('Next State')
    axes[0].set_ylabel('Current State')
    
    # Progression transitions
    sns.heatmap(mc_prog.P, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                xticklabels=mc_prog.class_names, yticklabels=mc_prog.class_names)
    axes[1].set_title('Progression Transitions')
    axes[1].set_xlabel('Next State')
    axes[1].set_ylabel('Current State')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Markov transitions saved to {save_path}")
    else:
        plt.show()


def plot_hmm_structure(hmm_model, task, save_path=None):
    """Plot HMM structure (emission and transition matrices)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transition matrix
    n_hidden = hmm_model.model.transmat_.shape[0]
    sns.heatmap(hmm_model.model.transmat_[:min(8, n_hidden), :min(8, n_hidden)], 
                annot=True, fmt='.3f', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'HMM Transition Matrix (first 8 states)')
    axes[0].set_xlabel('Next Hidden State')
    axes[0].set_ylabel('Current Hidden State')
    
    # Emission matrix
    n_obs = hmm_model.model.emissionprob_.shape[1]
    sns.heatmap(hmm_model.model.emissionprob_[:min(8, n_hidden), :min(5, n_obs)], 
                annot=True, fmt='.3f', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'HMM Emission Matrix')
    axes[1].set_xlabel('Observation')
    axes[1].set_ylabel('Hidden State')
    
    plt.suptitle(f'HMM Structure - {task.upper()}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  HMM structure saved to {save_path}")
    else:
        plt.show()


def plot_rnn_training(lstm_model, gru_model, task, save_path=None):
    """Plot RNN training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(lstm_model.train_losses, label='LSTM Train', linewidth=2)
    axes[0].plot(lstm_model.val_losses, label='LSTM Val', linewidth=2, linestyle='--')
    axes[0].plot(gru_model.train_losses, label='GRU Train', linewidth=2)
    axes[0].plot(gru_model.val_losses, label='GRU Val', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Curves - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(lstm_model.train_accs, label='LSTM Train', linewidth=2)
    axes[1].plot(lstm_model.val_accs, label='LSTM Val', linewidth=2, linestyle='--')
    axes[1].plot(gru_model.train_accs, label='GRU Train', linewidth=2)
    axes[1].plot(gru_model.val_accs, label='GRU Val', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Curves - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'RNN Training - {task.upper()}', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  RNN training curves saved to {save_path}")
    else:
        plt.show()


def plot_model_comparison(all_metrics, save_path=None):
    """
    Plot model comparison bar chart.
    Handles dictionaries, tuples, and various input formats.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract metrics from various formats
    metrics_list = []
    
    for m in all_metrics:
        if isinstance(m, dict):
            # Dictionary format
            if 'task' in m and 'model' in m and 'accuracy' in m:
                metrics_list.append(m)
        elif isinstance(m, tuple) and len(m) >= 3:
            # Tuple format: (model_name, task, metrics_dict)
            if isinstance(m[2], dict) and 'accuracy' in m[2]:
                metrics_list.append({
                    'model': m[0],
                    'task': m[1],
                    'accuracy': m[2]['accuracy']
                })
        elif hasattr(m, 'get') and callable(m.get):
            # Object with get method
            if m.get('task') and m.get('model') and m.get('accuracy'):
                metrics_list.append(m)
    
    if not metrics_list:
        print("  Warning: No valid metrics found for model comparison plot")
        return
    
    # Group by task
    tasks = []
    for m in metrics_list:
        if m['task'] not in tasks:
            tasks.append(m['task'])
    
    models = []
    for m in metrics_list:
        if m['model'] not in models:
            models.append(m['model'])
    
    if not tasks:
        print("  Warning: No tasks found in metrics")
        return
    
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]
    
    for idx, task in enumerate(tasks):
        task_metrics = [m for m in metrics_list if m.get("task") == task]
        task_models = [m["model"] for m in task_metrics]
        task_accs = [m["accuracy"] for m in task_metrics]
        
        # Sort by accuracy
        sorted_pairs = sorted(zip(task_accs, task_models), reverse=True)
        if sorted_pairs:
            task_accs, task_models = zip(*sorted_pairs)
            
            bars = axes[idx].bar(range(len(task_accs)), task_accs, 
                                color='steelblue', edgecolor='black')
            axes[idx].set_xticks(range(len(task_accs)))
            axes[idx].set_xticklabels(task_models, rotation=45, ha='right')
            axes[idx].set_ylabel('Accuracy')
            axes[idx].set_title(f'Task: {task.upper()}')
            axes[idx].set_ylim(0, 1.05)
            
            # Add value labels on bars
            for bar, acc in zip(bars, task_accs):
                axes[idx].text(bar.get_x() + bar.get_width()/2, 
                              bar.get_height() + 0.01,
                              f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Model comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_per_class_f1(all_results, save_path=None):
    """
    Plot per-class F1 scores heatmap.
    
    Parameters:
    -----------
    all_results : list of tuples (model_name, task, y_true, y_pred)
    save_path : str, optional
    """
    from sklearn.metrics import f1_score
    
    # Organize results by task
    tasks = {}
    for model_name, task, y_true, y_pred in all_results:
        if task not in tasks:
            tasks[task] = {}
        tasks[task][model_name] = (y_true, y_pred)
    
    for task, models in tasks.items():
        # Get unique classes
        all_classes = set()
        for model_name, (y_true, y_pred) in models.items():
            all_classes.update(y_true)
            all_classes.update(y_pred)
        classes = sorted(all_classes)
        
        # Build F1 matrix
        f1_matrix = []
        model_names = []
        for model_name, (y_true, y_pred) in models.items():
            model_names.append(model_name)
            f1_scores = f1_score(y_true, y_pred, labels=classes, 
                                 average=None, zero_division=0)
            f1_matrix.append(f1_scores)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=classes, yticklabels=model_names)
        plt.title(f'Per-Class F1 Scores - {task.upper()}')
        plt.xlabel('Class')
        plt.ylabel('Model')
        plt.tight_layout()
        
        if save_path:
            # Add task to filename
            base, ext = os.path.splitext(save_path)
            task_path = f"{base}_{task}{ext}"
            plt.savefig(task_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Per-class F1 plot saved to {task_path}")
        else:
            plt.show()


def plot_sample_trajectories(test_patients, mc_model, hmm_model, 
                             lstm_model, gru_model, n_samples=5, 
                             max_len=10, save_path=None):
    """
    Plot sample patient trajectories with model predictions.
    """
    import matplotlib.pyplot as plt
    
    n_samples = min(n_samples, len(test_patients))
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for idx in range(n_samples):
        patient = test_patients[idx]
        true_seq = patient["diag_seq"][:max_len]
        years = patient["years"][:max_len]
        
        # Generate predictions from each model
        mc_preds = []
        hmm_preds = []
        
        # Markov Chain predictions
        for t in range(1, len(true_seq)):
            mc_preds.append(mc_model.predict_next(true_seq[t-1]))
        
        # HMM predictions
        for t in range(1, len(true_seq)):
            hmm_preds.append(hmm_model.predict_next_obs(true_seq[:t]))
        
        # Plot
        axes[idx].plot(years, true_seq, 'ko-', label='True', linewidth=2, markersize=8)
        
        if len(mc_preds) == len(years[1:]):
            axes[idx].plot(years[1:], mc_preds, 'bs-', label='Markov', 
                          linewidth=1.5, markersize=6, alpha=0.7)
        
        if len(hmm_preds) == len(years[1:]):
            axes[idx].plot(years[1:], hmm_preds, 'g^-', label='HMM', 
                          linewidth=1.5, markersize=6, alpha=0.7)
        
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Diagnosis (0=B, 1=M)')
        axes[idx].set_title(f'Patient {idx+1}: {patient["id"]}')
        axes[idx].set_yticks([0, 1])
        axes[idx].set_yticklabels(['Benign', 'Malignant'])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Sample trajectories saved to {save_path}")
    else:
        plt.show()