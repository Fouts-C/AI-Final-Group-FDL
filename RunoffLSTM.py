"""
LSTM Post-Processing of National Water Model (NWM) Streamflow Forecasts

Builds, trains, and evaluates an LSTM that learns NWM forecast errors
(residuals) and subtracts the predicted error from raw forecasts to
produce corrected streamflow estimates.

References
[1] Han, S. & Morrison, R.R. (2022). Improved runoff forecasting performance
    through error correction with a deep learning approach. Journal of
    Hydrology, 608, 127653.
    https://doi.org/10.1016/j.jhydrol.2022.127653
[2] Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M.
    (2018). Rainfall-runoff modelling using Long Short-Term Memory (LSTM)
    networks. Hydrol. Earth Syst. Sci., 22, 6005-6022.
    https://doi.org/10.5194/hess-22-6005-2018
[3] Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory.
    Neural Computation, 9(8), 1735-1780.
[4] PyTorch LSTM Documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
    JMLR, 12, 2825-2830.  (StandardScaler)

Data
----
NWM v2.1 hourly forecasts (1-18 h lead) and USGS observed streamflow at
gauges 09520500 (streamID 20380357) and 11266500 (streamID 21609641).
Preprocessed in RunoffPreprocessing.ipynb -> data/processed/
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOOKBACK    = 24         # hours of history fed to the LSTM
BATCH_SIZE  = 512
HIDDEN_SIZE = 64         # LSTM hidden units per layer
NUM_LAYERS  = 2          # stacked LSTM layers
DROPOUT     = 0.2        # dropout between LSTM layers
LR          = 1e-3       # Adam initial learning rate
EPOCHS      = 50
PATIENCE    = 10         # early-stopping patience (epochs w/o val improvement)
VAL_FRAC    = 0.15       # last fraction of training period used for validation
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED        = 42
OUT_DIR     = "pics"
MODEL_PATH  = "lstm_runoff.pt"

FEAT_COLS = [
    "nwm_forecast", "residual", "lead_time_hrs",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
N_FEATURES = len(FEAT_COLS)
GAGE_MAP   = {20380357: "09520500", 21609641: "11266500"}

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device  : {DEVICE}")
print(f"Lookback: {LOOKBACK}   Batch: {BATCH_SIZE}   Hidden: {HIDDEN_SIZE}")
print(f"Layers  : {NUM_LAYERS}   Dropout: {DROPOUT}   LR: {LR}\n")


# 1.1  Load preprocessed data
train_raw = pd.read_csv(
    "data/processed/train_val.csv",
    parse_dates=["model_initialization_time", "model_output_valid_time"],
)
test_raw = pd.read_csv(
    "data/processed/test.csv",
    parse_dates=["model_initialization_time", "model_output_valid_time"],
)
print(f"Loaded — Train/Val: {len(train_raw):,}   Test: {len(test_raw):,}")


# 1.2  Cyclical temporal features
def add_temporal(df):
    """Encode hour-of-day and day-of-year as sin/cos pairs."""
    df = df.copy()
    h = df["model_output_valid_time"].dt.hour
    d = df["model_output_valid_time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * d / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * d / 365.25)
    return df

train_raw = add_temporal(train_raw)
test_raw  = add_temporal(test_raw)


# 1.3  Fit scalers on training data only
feat_scaler = StandardScaler().fit(train_raw[FEAT_COLS])
tgt_scaler  = StandardScaler().fit(train_raw[["residual"]])

train_sc = train_raw.copy()
test_sc  = test_raw.copy()
train_sc[FEAT_COLS] = feat_scaler.transform(train_raw[FEAT_COLS])
test_sc[FEAT_COLS]  = feat_scaler.transform(test_raw[FEAT_COLS])


#  1.4  Concatenate & assign split labels
full_sc  = pd.concat([train_sc, test_sc], ignore_index=True)
full_raw = pd.concat([train_raw, test_raw], ignore_index=True)

n_train = len(train_sc)
train_max = train_raw["model_output_valid_time"].max()
train_min = train_raw["model_output_valid_time"].min()
val_cutoff = train_max - (train_max - train_min) * VAL_FRAC

is_orig_train = np.arange(len(full_sc)) < n_train
is_after_cutoff = full_raw["model_output_valid_time"].values > val_cutoff

split_labels = np.full(len(full_sc), "train", dtype=object)
split_labels[~is_orig_train] = "test"
split_labels[is_orig_train & is_after_cutoff] = "val"
full_sc["_split"] = split_labels

for s in ("train", "val", "test"):
    print(f"  {s:5s}: {(split_labels == s).sum():>8,} rows")

# 1.5  Build sliding-window sequences
def build_sequences(sc_df, raw_df, lookback):
    Xs = {s: [] for s in ("train", "val", "test")}
    ys = {s: [] for s in ("train", "val", "test")}
    meta_parts = []

    for (sid, lt), grp_idx in sc_df.groupby(["streamID", "lead_time_hrs"]).groups.items():
        grp = sc_df.loc[sorted(grp_idx)].sort_values("model_output_valid_time")
        raw = raw_df.loc[grp.index]

        feats   = grp[FEAT_COLS].values.astype(np.float32)
        targets = tgt_scaler.transform(
            raw[["residual"]].values
        ).flatten().astype(np.float32)
        splits = grp["_split"].values
        n = len(feats)

        if n <= lookback:
            continue

        # Vectorised window creation (no Python loop over timesteps)
        # sliding_window_view puts the window axis last → (n-lb+1, F, lb)
        # transpose to (n-lb+1, lb, F) so LSTM sees (batch, seq_len, features)
        windows = sliding_window_view(feats, lookback, axis=0).transpose(0, 2, 1)
        X_all = windows[: n - lookback]   # (n-lb, lb, F)
        y_all = targets[lookback:]         # (n-lb,)
        s_all = splits[lookback:]          # (n-lb,)

        for s in ("train", "val", "test"):
            mask = s_all == s
            if mask.any():
                Xs[s].append(X_all[mask].copy())
                ys[s].append(y_all[mask].copy())

        # Keep unscaled metadata for test evaluation later
        t_mask = s_all == "test"
        if t_mask.any():
            raw_tail = raw.iloc[lookback:].reset_index(drop=True)
            raw_test = raw_tail[t_mask].reset_index(drop=True)
            meta_parts.append(pd.DataFrame({
                "streamID":      sid,
                "lead_time_hrs": lt,
                "valid_time":    raw_test["model_output_valid_time"].values,
                "raw_nwm":       raw_test["nwm_forecast"].values,
                "raw_obs":       raw_test["usgs_observed"].values,
                "raw_res":       raw_test["residual"].values,
            }))

    out = {}
    for s in ("train", "val", "test"):
        out[f"X_{s}"] = np.concatenate(Xs[s]) if Xs[s] else np.empty((0, lookback, N_FEATURES), dtype=np.float32)
        out[f"y_{s}"] = np.concatenate(ys[s]) if ys[s] else np.empty(0, dtype=np.float32)
    out["test_meta"] = pd.concat(meta_parts, ignore_index=True) if meta_parts else pd.DataFrame()
    return out


print("\nBuilding sequences … ", end="", flush=True)
t0 = time.time()
seq = build_sequences(full_sc, full_raw, LOOKBACK)
print(f"done ({time.time() - t0:.1f}s)")

for s in ("train", "val", "test"):
    print(f"  {s:5s}  X {seq[f'X_{s}'].shape}   y {seq[f'y_{s}'].shape}")

#  Model Architecture
class SequenceDataset(Dataset):
    """Wraps numpy arrays as a PyTorch Dataset."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class RunoffLSTM(nn.Module):
    """
    Stacked LSTM followed by a small MLP head.

    Input  : (batch, lookback, n_features)
    Output : (batch,)  — predicted residual (scaled)
    """
    def __init__(self, n_features, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                      # (B, T, H)
        return self.head(out[:, -1, :]).squeeze(-1) # (B,)


# DataLoaders
train_ds = SequenceDataset(seq["X_train"], seq["y_train"])
val_ds   = SequenceDataset(seq["X_val"],   seq["y_val"])
test_ds  = SequenceDataset(seq["X_test"],  seq["y_test"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2)

model = RunoffLSTM(N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,}")
print(model)

