import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RNN_HIDDEN_DIM, RNN_N_LAYERS, RNN_DROPOUT, RNN_LR,
    RNN_EPOCHS, RNN_BATCH_SIZE, SEED, CKPT_DIR,
    DIAGNOSIS_CLASSES, PROGRESSION_CLASSES, FEATURE_COLS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Don't set N_FEATURES here - will detect from data
CLASS_NAMES = {
    "diag": DIAGNOSIS_CLASSES,
    "prog": PROGRESSION_CLASSES,
}
N_CLASSES = {
    "diag": 2,
    "prog": 2,
}


class Attention(nn.Module):
    """Attention mechanism for RNN"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        attention_weights = torch.softmax(self.attention(rnn_outputs), dim=1)
        context = torch.sum(attention_weights * rnn_outputs, dim=1)
        return context, attention_weights


class SequenceDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceNet(nn.Module):
    
    def __init__(self, cell="lstm", n_features=30, n_classes=2,
                 hidden_dim=RNN_HIDDEN_DIM, n_layers=RNN_N_LAYERS,
                 dropout=RNN_DROPOUT, bidirectional=True):
        super().__init__()
        self.cell_type = cell.lower()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.attention = Attention(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, n_classes)

    def forward(self, x):
        if self.cell_type == "lstm":
            out, (h_n, _) = self.rnn(x)
        else:
            out, h_n = self.rnn(x)
        
        # Apply attention
        context, attention_weights = self.attention(out)
        context = self.dropout(context)
        return self.classifier(context)


class RNNModel:

    def __init__(self, cell="lstm", task="diag", n_classes=2,
                 hidden_dim=RNN_HIDDEN_DIM, n_layers=RNN_N_LAYERS,
                 dropout=RNN_DROPOUT, lr=RNN_LR, seed=SEED):
        torch.manual_seed(seed)
        self.cell = cell.lower()
        self.task = task
        self.n_classes = n_classes
        self.class_names = CLASS_NAMES[task]
        self.net = None  # Will be created during fit when we know n_features
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Store architecture params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr

    def _create_network(self, n_features):
        self.net = SequenceNet(
            cell=self.cell, 
            n_features=n_features, 
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim, 
            n_layers=self.n_layers, 
            dropout=self.dropout,
            bidirectional=True
        ).to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE, verbose=True):
        
        # Get input dimension from data
        n_features = X_train.shape[2]
        if self.net is None:
            self._create_network(n_features)
            print(f"  Created network with {n_features} input features")

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            SequenceDataset(X_val, y_val),
            batch_size=batch_size
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience = 10

        for epoch in range(1, epochs + 1):
            # Training
            self.net.train()
            t_loss, t_correct, t_total = 0.0, 0, 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.net(Xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                t_loss += loss.item() * len(yb)
                t_correct += (logits.argmax(1) == yb).sum().item()
                t_total += len(yb)

            # Validation
            self.net.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    logits = self.net(Xb)
                    loss = self.criterion(logits, yb)
                    v_loss += loss.item() * len(yb)
                    v_correct += (logits.argmax(1) == yb).sum().item()
                    v_total += len(yb)

            avg_tl = t_loss / t_total
            avg_vl = v_loss / v_total
            avg_ta = t_correct / t_total
            avg_va = v_correct / v_total

            self.train_losses.append(avg_tl)
            self.val_losses.append(avg_vl)
            self.train_accs.append(avg_ta)
            self.val_accs.append(avg_va)
            self.scheduler.step(avg_vl)

            # Early stopping
            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

            if verbose and epoch % 10 == 0:
                print(f"  [{self.cell.upper():4s}|{self.task}] "
                      f"Epoch {epoch:3d}/{epochs}  "
                      f"Train loss:{avg_tl:.4f} acc:{avg_ta:.3f}  "
                      f"Val loss:{avg_vl:.4f} acc:{avg_va:.3f}")

        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.net(Xt)
        return logits.argmax(1).cpu().numpy()

    def predict_proba(self, X):
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.net(Xt)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def log_likelihood(self, X, y):
        probs = self.predict_proba(X)
        return sum(np.log(probs[i, y[i]] + 1e-10) for i in range(len(y)))

    def save(self, path=None):
        if self.net is None:
            print("  Model not trained yet, nothing to save")
            return
        path = path or os.path.join(CKPT_DIR, f"{self.cell}_{self.task}.pt")
        torch.save(self.net.state_dict(), path)
        print(f"  Checkpoint saved → {path}")

    def load(self, path):
        # Need to know n_features first - load from checkpoint
        print("  Loading requires n_features - use load_with_features()")
        
    def load_with_features(self, path, n_features):
        self._create_network(n_features)
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.net.eval()
        return self


if __name__ == "__main__":
    from utils.preprocessor import load_and_prepare
    from sklearn.metrics import classification_report
    from config import DIAGNOSIS_CLASSES

    data = load_and_prepare()
    d = data["rnn_diag"]

    for cell in ["lstm", "gru"]:
        print(f"\n{'='*55}\n  {cell.upper()} — Task: diag\n{'='*55}")
        model = RNNModel(cell=cell, task="diag")
        model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], verbose=True)
        y_pred = model.predict(d["X_test"])

        print(classification_report(
            d["y_test"], y_pred,
            target_names=DIAGNOSIS_CLASSES, 
            zero_division=0
        ))