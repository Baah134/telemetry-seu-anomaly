"""Forecasters. Tiny MLP (main, MCU-sized) and LSTM reference (telemanom-style).

TinyMLP: [W values + n_cmd commands] -> 32 -> 16 -> 1, ReLU. An MLP (not a CNN) keeps
the integer-only reference and the fault study transparent: three int8 GEMVs.
Params = (W + n_cmd)*32 + 32 + 32*16 + 16 + 16 + 1  (SMAP n_cmd=24: 3,393; MSL n_cmd
up to 54: 4,353) -> <= 5 KB as int8.

LSTMRef: 2 x 80 LSTM over the window (value + commands per step), linear head.
telemanom used 2 x 80 with a 250-step window; we use the same 64-step window as the
tiny model so the comparison isolates model capacity.
"""
import numpy as np
import torch
from torch import nn

H1, H2 = 32, 16


class TinyMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, 1)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))).squeeze(-1)


class LSTMRef(nn.Module):
    def __init__(self, w, n_cmd, hidden=80):
        super().__init__()
        self.w, self.n_cmd = w, n_cmd
        self.lstm = nn.LSTM(1 + n_cmd, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        v = x[:, :self.w].unsqueeze(-1)
        c = x[:, self.w:].unsqueeze(1).expand(-1, self.w, -1)
        h, _ = self.lstm(torch.cat([v, c], dim=-1))
        return self.head(h[:, -1]).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_forecaster(model, Xtr, ytr, Xva, yva, seed, epochs=30, batch=256, lr=3e-3,
                     patience=5):
    """Adam + MSE with early stopping on the validation split (train data only)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.mean((model(Xt[b]) - yt[b]) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(torch.mean((model(Xv) - yv) ** 2)) if len(Xv) else float(loss)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def predict(model, X, batch=4096):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(X[i:i + batch])).numpy()
                               for i in range(0, len(X), batch)]) if len(X) else np.zeros(0)
