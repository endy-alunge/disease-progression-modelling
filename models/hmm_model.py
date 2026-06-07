import os, sys, warnings
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from hmmlearn import hmm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    HMM_N_HIDDEN, HMM_N_ITER, HMM_TOL, SEED,
    DIAGNOSIS_CLASSES, PROGRESSION_CLASSES, STAGE_CLASSES
)

CLASS_NAMES = {
    "diag": DIAGNOSIS_CLASSES,
    "prog": PROGRESSION_CLASSES,
    "stage": STAGE_CLASSES,
}

N_OBS = {
    "diag": 2,
    "prog": 2,
    "stage": 5,
}


class HMMModel:

    def __init__(self, task="diag", n_hidden=HMM_N_HIDDEN,
                 n_iter=HMM_N_ITER, tol=HMM_TOL, seed=SEED):
        self.task = task
        self.n_hidden = n_hidden
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.model = None
        self.class_names = CLASS_NAMES[task]
        self.n_obs = N_OBS[task]
        
        # For stage task, use more hidden states
        if task == "stage":
            self.n_hidden = max(n_hidden, 8)

    def _prepare(self, sequences):
        obs = np.concatenate(sequences).reshape(-1, 1).astype(int)
        lengths = [len(s) for s in sequences]
        return obs, lengths

    def _init_transmat(self):
        transmat = np.eye(self.n_hidden) * 0.7
        transmat = transmat + np.ones((self.n_hidden, self.n_hidden)) * 0.3 / self.n_hidden
        return transmat / transmat.sum(axis=1, keepdims=True)

    def fit(self, sequences):
        obs, lengths = self._prepare(sequences)
        
        # Try multiple restarts for better convergence
        best_model = None
        best_score = -np.inf
        
        for restart in range(5):  # 5 restarts
            try:
                model = hmm.CategoricalHMM(
                    n_components=self.n_hidden,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.seed + restart,
                    verbose=False,
                )
                # Initialize with diagonal transition matrix
                model.transmat_ = self._init_transmat()
                model.startprob_ = np.ones(self.n_hidden) / self.n_hidden
                
                model.fit(obs, lengths)
                score = model.score(obs, lengths)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                continue
        
        self.model = best_model
        return self

    def decode(self, sequence):
        obs = np.array(sequence).reshape(-1, 1)
        return self.model.decode(obs, algorithm="viterbi")

    def predict_next_obs(self, history):
        if len(history) == 0:
            return int(np.argmax(self.model.emissionprob_.mean(axis=0)))
        
        try:
            obs = np.array(history).reshape(-1, 1)
            _, hidden_seq = self.model.decode(obs, algorithm="viterbi")
            last_h = hidden_seq[-1]
            next_h = int(np.argmax(self.model.transmat_[last_h]))
            return int(np.argmax(self.model.emissionprob_[next_h]))
        except Exception:
            return int(np.argmax(self.model.emissionprob_.mean(axis=0)))

    def predict_sequence(self, sequences):
        y_true, y_pred = [], []
        for seq in sequences:
            for t in range(1, len(seq)):
                y_true.append(seq[t])
                y_pred.append(self.predict_next_obs(seq[:t]))
        return np.array(y_true), np.array(y_pred)

    def log_likelihood(self, sequences):
        obs, lengths = self._prepare(sequences)
        return self.model.score(obs, lengths)

    def print_structure(self):
        print(f"\n  Hidden states: {self.n_hidden}")
        print(f"  Observations:  {self.class_names}")
        print("\n  Transition matrix (first 5 hidden states):")
        for i in range(min(5, self.n_hidden)):
            row = "  ".join(f"{v:.3f}" for v in self.model.transmat_[i][:5])
            print(f"    H{i}: [{row}...]")
        print("\n  Emission matrix (first 5 hidden states):")
        header = "    ".join(f"{c:>8}" for c in self.class_names[:5])
        print(f"          {header}")
        for i in range(min(5, self.n_hidden)):
            row = "    ".join(f"{v:>8.3f}" for v in self.model.emissionprob_[i][:self.n_obs])
            print(f"    H{i}: [{row}]")

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
        return y_true, y_pred, {"accuracy": acc, "log_likelihood": ll}
    
    @property
    def is_trained(self):
        return self.model is not None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.preprocessor import load_and_prepare
    data = load_and_prepare()

    print(f"\n{'='*55}\n  HMM — Task: diag\n{'='*55}")
    model = HMMModel(task="diag", n_hidden=4)
    model.fit(data["diag_seqs"]["train"])
    model.print_structure()
    model.evaluate(data["diag_seqs"]["test"])
    
    # Test stage task if available
    if "stage_seqs" in data:
        print(f"\n{'='*55}\n  HMM — Task: stage (5 classes)\n{'='*55}")
        model_stage = HMMModel(task="stage", n_hidden=8)
        model_stage.fit(data["stage_seqs"]["train"])
        model_stage.print_structure()
        model_stage.evaluate(data["stage_seqs"]["test"])