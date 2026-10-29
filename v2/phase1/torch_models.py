"""
Sentinel Credit v2, Phase 1 — PyTorch models (V2-02).

Plain English:
- MLPClassifierTorch: a small neural network (a multilayer perceptron, MLP)
  that predicts default from the 13 features. It behaves like a scikit-learn
  model (fit / predict_proba), so the SAME Platt calibration the champion used
  (CalibratedClassifierCV, sigmoid, 5 folds) can wrap it unchanged.
- train_autoencoder: a network that learns to compress and rebuild the
  features of customers who did NOT default. A customer it rebuilds badly
  looks unusual (high reconstruction error); Phase 1 reports whether unusual
  customers default more.

What the code does: standardises inputs on the data it is fitted on, trains
with Adam, weights the rare default class (like XGBoost's scale_pos_weight),
and stops early when a held-back slice of that same data stops improving.
Seeds are fixed so runs repeat.
"""

import copy

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)        # a logit (raw score), not a probability


class MLPClassifierTorch(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(64, 32), dropout=0.2, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=256, max_epochs=100, validation_fraction=0.15, patience=8, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        _seed(self.random_state)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_state)
        self.scaler_ = StandardScaler().fit(X_fit)
        xf = torch.tensor(self.scaler_.transform(X_fit), dtype=torch.float32)
        xv = torch.tensor(self.scaler_.transform(X_val), dtype=torch.float32)
        yf, yv = torch.tensor(y_fit), torch.tensor(y_val)
        pos_weight = torch.tensor((len(y_fit) - y_fit.sum()) / max(y_fit.sum(), 1.0))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model_ = MLP(X.shape[1], list(self.hidden), self.dropout)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.random_state)
        best, best_state, wait, self.epochs_ = np.inf, None, 0, 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for idx in torch.randperm(len(xf), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(xf[idx]), yf[idx]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val = loss_fn(self.model_(xv), yv).item()
            if val < best - 1e-5:
                best, best_state, wait, self.epochs_ = val, copy.deepcopy(self.model_.state_dict()), 0, epoch + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_val_loss_ = best
        return self

    def decision_function(self, X):
        x = torch.tensor(self.scaler_.transform(np.asarray(X, dtype=np.float32)), dtype=torch.float32)
        with torch.no_grad():
            return self.model_(x).numpy()

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AutoEncoder(nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        enc, prev = [], n_in
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec = []
        for h in list(reversed(hidden))[1:] + [n_in]:
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec = dec[:-1]                         # linear output layer
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_normal, hidden, learning_rate, batch_size, max_epochs, validation_fraction,
                      patience, random_state=42):
    """Fit on 'normal' rows only. Returns (scaler, model, epochs, best_val_mse)."""
    _seed(random_state)
    X_normal = np.asarray(X_normal, dtype=np.float32)
    X_fit, X_val = train_test_split(X_normal, test_size=validation_fraction, random_state=random_state)
    scaler = StandardScaler().fit(X_fit)
    xf = torch.tensor(scaler.transform(X_fit), dtype=torch.float32)
    xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    model = AutoEncoder(X_normal.shape[1], list(hidden))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(random_state)
    best, best_state, wait, epochs = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(xf), generator=gen).split(batch_size):
            opt.zero_grad()
            loss_fn(model(xf[idx]), xf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), xv).item()
        if val < best - 1e-6:
            best, best_state, wait, epochs = val, copy.deepcopy(model.state_dict()), 0, epoch + 1
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return scaler, model, epochs, best


def reconstruction_error(scaler, model, X):
    x = torch.tensor(scaler.transform(np.asarray(X, dtype=np.float32)), dtype=torch.float32)
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=1).numpy()
