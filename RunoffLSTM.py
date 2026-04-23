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

    # Group on raw_df so `lt` is the unscaled hour value (FEAT_COLS includes
    # lead_time_hrs, which gets z-scored — we need the original for metadata).
    for (sid, lt), grp_idx in raw_df.groupby(["streamID", "lead_time_hrs"]).groups.items():
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


# Training Loop
# Adam + MSELoss on scaled residuals (Han & Morrison 2022; Kratzert et al. 2018)
# ReduceLROnPlateau halves LR when val loss stalls; early stop after PATIENCE
# epochs without improvement. Gradient clipping (max_norm=1.0) stabilizes the
# stacked LSTM per Kratzert et al. (2018) §3.2.
from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)


def run_epoch(loader, train_mode):
    """Single pass over a DataLoader. Returns mean MSE over all samples."""
    model.train(train_mode)
    total_loss, total_n = 0.0, 0
    torch.set_grad_enabled(train_mode)
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        preds = model(X)
        loss = criterion(preds, y)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_n += X.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total_n


history = {"train_loss": [], "val_loss": [], "lr": []}
best_val = float("inf")
epochs_no_improve = 0

# Skip training if a checkpoint already exists (set FORCE_RETRAIN=1 to override)
if os.path.exists(MODEL_PATH) and os.environ.get("FORCE_RETRAIN") != "1":
    print(f"\nFound existing checkpoint {MODEL_PATH} — skipping training.")
    print("Set FORCE_RETRAIN=1 to retrain from scratch.")
    # Minimal placeholder history so the loss-curve plot still renders something
    history["train_loss"] = [float("nan")]
    history["val_loss"]   = [float("nan")]
    history["lr"]         = [LR]
    best_val = float("nan")
else:
    print("\nTraining …")
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(train_loader, train_mode=True)
        val_loss   = run_epoch(val_loader,   train_mode=False)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1

        tag = "  *best*" if improved else ""
        print(
            f"  epoch {epoch:3d}/{EPOCHS}  "
            f"train {train_loss:.4f}   val {val_loss:.4f}   "
            f"lr {current_lr:.2e}   ({time.time() - t0:.1f}s){tag}"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"  early stop — no val improvement for {PATIENCE} epochs")
            break

    print(f"Training done in {time.time() - t_start:.1f}s  |  best val MSE {best_val:.4f}")


# Test-set Evaluation
# Reload best weights, predict residuals, inverse-transform to cfs, form
# corrected = raw_nwm - pred_residual, compare vs raw NWM with RMSE/MAE/NSE
# (Han & Morrison 2022 Table 2; Kratzert et al. 2018). NSE is the headline
# metric — it normalizes by observed variance so results compare across basins.
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

preds_scaled = []
with torch.no_grad():
    for X, _ in test_loader:
        preds_scaled.append(model(X.to(DEVICE)).cpu().numpy())
preds_scaled = np.concatenate(preds_scaled)

# Inverse-transform to cfs (sklearn StandardScaler docs)
pred_residual = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

results = seq["test_meta"].copy()
results["pred_residual"] = pred_residual
results["corrected"]     = results["raw_nwm"] - results["pred_residual"]


def rmse(obs, pred):
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def mae(obs, pred):
    return float(np.mean(np.abs(obs - pred)))

def nse(obs, pred):
    """Nash-Sutcliffe Efficiency. 1.0 = perfect, 0 = no better than mean."""
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)

def summarize(df, label):
    obs = df["raw_obs"].values
    return {
        "group":          label,
        "n":              len(df),
        "rmse_raw":       rmse(obs, df["raw_nwm"].values),
        "rmse_corrected": rmse(obs, df["corrected"].values),
        "mae_raw":        mae(obs,  df["raw_nwm"].values),
        "mae_corrected":  mae(obs,  df["corrected"].values),
        "nse_raw":        nse(obs,  df["raw_nwm"].values),
        "nse_corrected":  nse(obs,  df["corrected"].values),
    }


rows = [summarize(results, "ALL")]
for sid, grp in results.groupby("streamID"):
    rows.append(summarize(grp, f"station {GAGE_MAP[sid]}"))

per_lead_rows = []
for (sid, lt), grp in results.groupby(["streamID", "lead_time_hrs"]):
    lt_i = int(lt)
    r = summarize(grp, f"{GAGE_MAP[sid]} lead {lt_i:>2d}h")
    r["streamID"], r["lead_time_hrs"] = sid, lt_i
    per_lead_rows.append(r)

metrics_overall = pd.DataFrame(rows)
metrics_perlead = pd.DataFrame(per_lead_rows)

print("\n=== Test Metrics — Overall / Per-Station ===")
print(metrics_overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== Test Metrics — Per Station × Lead Time ===")
print(
    metrics_perlead[[
        "group", "n", "rmse_raw", "rmse_corrected",
        "mae_raw", "mae_corrected", "nse_raw", "nse_corrected",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

metrics_overall.to_csv(os.path.join(OUT_DIR, "metrics_overall.csv"), index=False)
metrics_perlead.to_csv(os.path.join(OUT_DIR, "metrics_per_lead.csv"), index=False)


# Plots: loss curves, RMSE by lead, storm-event time series, scatter
if len(history["train_loss"]) > 1 and not np.isnan(history["train_loss"][0]):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled residual)")
    ax.set_title("Training / Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150)
    plt.close(fig)
else:
    print("  (skipping loss_curves.png — training was skipped)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (sid, grp) in zip(axes, metrics_perlead.groupby("streamID")):
    grp = grp.sort_values("lead_time_hrs")
    ax.plot(grp["lead_time_hrs"], grp["rmse_raw"],       "o-", label="raw NWM")
    ax.plot(grp["lead_time_hrs"], grp["rmse_corrected"], "s-", label="LSTM-corrected")
    ax.set_title(f"Station {GAGE_MAP[sid]}")
    ax.set_xlabel("Lead time (h)")
    ax.grid(alpha=0.3); ax.legend()
axes[0].set_ylabel("RMSE (cfs)")
fig.suptitle("RMSE vs Lead Time — corrected vs raw NWM")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rmse_by_lead.png"), dpi=150)
plt.close(fig)

# Highest-flow 14-day window per station (Han & Morrison §4.3)
LEADS_TO_PLOT = [1, 6, 12, 18]
fig, axes = plt.subplots(len(GAGE_MAP), 1, figsize=(12, 4 * len(GAGE_MAP)))
if len(GAGE_MAP) == 1:
    axes = [axes]
for ax, (sid, name) in zip(axes, GAGE_MAP.items()):
    st = results[results["streamID"] == sid].copy()
    st["valid_time"] = pd.to_datetime(st["valid_time"])
    daily_mean = st.groupby(st["valid_time"].dt.date)["raw_obs"].mean()
    peak_day = pd.Timestamp(daily_mean.idxmax())
    t0 = peak_day - pd.Timedelta(days=7)
    t1 = peak_day + pd.Timedelta(days=7)
    window = st[(st["valid_time"] >= t0) & (st["valid_time"] <= t1)]

    w1 = window[window["lead_time_hrs"] == 1].sort_values("valid_time")
    ax.plot(w1["valid_time"], w1["raw_obs"], "k-", lw=2, label="observed")
    ax.plot(w1["valid_time"], w1["raw_nwm"], color="tab:orange", lw=1.2, label="raw NWM (lead 1h)")
    for lt in LEADS_TO_PLOT:
        wl = window[window["lead_time_hrs"] == lt].sort_values("valid_time")
        ax.plot(wl["valid_time"], wl["corrected"], "--", lw=1, label=f"corrected lead {lt}h")
    ax.set_title(f"Station {name} — peak-flow window around {peak_day.date()}")
    ax.set_ylabel("Streamflow (cfs)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "timeseries.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (sid, grp) in zip(axes, results.groupby("streamID")):
    # Axis limits based on observed range (+ small margin) so raw-NWM outliers
    # don't blow up the view at the dry gauge. Outlier scatter still shows.
    obs_max = grp["raw_obs"].max()
    lim = obs_max * 1.1 if obs_max > 0 else 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.scatter(grp["raw_obs"], grp["raw_nwm"],   s=4, alpha=0.3, label="raw NWM")
    ax.scatter(grp["raw_obs"], grp["corrected"], s=4, alpha=0.3, label="corrected")
    ax.set_xlim(0, lim); ax.set_ylim(-lim * 0.1, lim)
    ax.set_xlabel("Observed (cfs)"); ax.set_ylabel("Predicted (cfs)")
    ax.set_title(f"Station {GAGE_MAP[sid]}")
    ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "scatter.png"), dpi=150)
plt.close(fig)

print(f"\nPlots saved to {OUT_DIR}/")
print("  loss_curves.png, rmse_by_lead.png, timeseries.png, scatter.png")
print("Done.")
