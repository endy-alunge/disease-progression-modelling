"""
models/markov_chain.py
-----------------------
First-order Markov Chain for longitudinal sequences.
Supports diagnosis, progression, and clinical stages.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DIAGNOSIS_CLASSES, PROGRESSION_CLASSES, 
    MC_SMOOTHING, STAGE_CLASSES
)

CLASS_NAMES = {
    "diag": DIAGNOSIS_CLASSES,
    "prog": PROGRESSION_CLASSES,
    "stage": STAGE_CLASSES,
}

N_STATES = {
    "diag": 2,
    "prog": 2,
    "stage": 5,
}


class MarkovChainModel:
    """
    First-order homogeneous Markov Chain.
    
    Supports:
      task='diag'  → states {0:B, 1:M}
      task='prog'  → states {0:Stable, 1:Progressed}
      task='stage' → states {0:Benign, 1:Stage I, 2:Stage II, 3:Stage III, 4:Stage IV}
    """

    def __init__(self, task="diag", smoothing=MC_SMOOTHING):
        self.task = task
        self.n_states = N_STATES[task]
        self.smoothing = smoothing
        self.counts = np.zeros((self.n_states, self.n_states))
        self.P = None
        self.class_names = CLASS_NAMES.get(task, [f"S{i}" for i in range(self.n_states)])

    def fit(self, sequences):
        """sequences: list of int lists"""
        self.counts[:] = 0.0
        for seq in sequences:
            for t in range(len(seq) - 1):
                i, j = seq[t], seq[t + 1]
                if 0 <= i < self.n_states and 0 <= j < self.n_states:
                    self.counts[i, j] += 1

        smooth = self.counts + self.smoothing
        self.P = smooth / smooth.sum(axis=1, keepdims=True)
        return self

    def predict_next(self, current_state):
        """Return most likely next state"""
        return int(np.argmax(self.P[current_state]))

    def predict_proba(self, current_state):
        return self.P[current_state].copy()

    def predict_sequence(self, sequences):
        """For each (state_t → state_t+1), predict next state."""
        y_true, y_pred = [], []
        for seq in sequences:
            for t in range(len(seq) - 1):
                y_true.append(seq[t + 1])
                y_pred.append(self.predict_next(seq[t]))
        return np.array(y_true), np.array(y_pred)

    def log_likelihood(self, sequences):
        ll = 0.0
        for seq in sequences:
            for t in range(len(seq) - 1):
                i, j = seq[t], seq[t + 1]
                ll += np.log(self.P[i, j] + 1e-10)
        return ll

    def print_transition_matrix(self):
        print(f"\n  Transition Matrix ({self.task.upper()}):")
        header = f"  {'':>12}" + "".join(f"{c:>12}" for c in self.class_names)
        print(header)
        print("  " + "-" * (12 + 12 * self.n_states))
        for i, ci in enumerate(self.class_names):
            row = f"  {ci:>12}" + "".join(f"{self.P[i,j]:>12.4f}" for j in range(self.n_states))
            print(row)

    def plot_transition_matrix(self, save_path=None):
        """Visualize transition matrix as heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.P, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Markov Chain Transition Matrix - {self.task.upper()}')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def evaluate(self, sequences, verbose=True):
        y_true, y_pred = self.predict_sequence(sequences)
        acc = accuracy_score(y_true, y_pred)
        ll = self.log_likelihood(sequences)
        
        if verbose:
            print(f"\n  Accuracy: {acc:.4f}   Log-Likelihood: {ll:.2f}")
            
            labels_to_print = list(range(len(self.class_names)))
            
            print(classification_report(
                y_true, y_pred,
                labels=labels_to_print,
                target_names=self.class_names, 
                zero_division=0
            ))
            
            # Print transition matrix for stage task
            if self.task == "stage":
                self.print_transition_matrix()
                
        return y_true, y_pred, {"accuracy": acc, "log_likelihood": ll}