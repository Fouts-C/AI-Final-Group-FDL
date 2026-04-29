Week of 4/6/2026 - 4/12/2026
Authors:
    - Josh Dula
    - Quinn Lautenslager
    - Carson Fouts

Tasks Completed:
    - Added all new runoff forecasting datasets (NWM forecasts + USGS observations for 2 stations) (Quinn Lautenslager)
    - Combined NWM streamflow CSVs and USGS observed CSVs into combined files in RunoffPreprocessing.ipynb (Quinn Lautenslager)
    - Added data preprocessing pipeline to RunoffPreprocessing.ipynb following zyBooks Ch. 5 data wrangling steps (Josh Dula)
    - Fixed USGS column mismatch between the two station files (Josh Dula)
    - Parsed datetime formats, computed lead times 1-18h, resampled USGS from 15-min to hourly (Josh Dula)
    - Cleaned missing values and duplicates, merged NWM + USGS, derived residual feature (Josh Dula)
    - Split data into train/val (Apr 2021 - Sep 2022) and test (Oct 2022 - Apr 2023) per project requirements (Josh Dula)
    - Saved processed datasets to data/processed/ (Josh Dula)
    - Added frame, JSON, and mask data for the hurricane damage dataset (Carson Fouts)
    - Added preprocessing script for the hurricane damage dataset (Carson Fouts)
    - Updated README.md and Update Log (Josh Dula, Carson Fouts)

Tasks In Progress (4/12/2026 - 4/19/2026):
    - Build LSTM model for runoff forecast error correction (Josh Dula, Quinn Lautenslager, Carson Fouts)
    - Evaluate model performance with RMSE, NSE metrics
    - Compare corrected forecasts vs original NWM vs observed USGS
    - Look into using PyTorch for the hurricane damage dataset (Carson Fouts)
    - Add data augmentation to prevent CNN overfitting (Carson Fouts)
    - Add train/test split for the hurricane damage dataset (Carson Fouts)

Week of 4/13/2026 - 4/19/2026
Authors:
    - Josh Dula
    - Carson Fouts

Tasks Completed:
    - Built LSTM feature engineering pipeline in RunoffLSTM.py (Phase 1) (Josh Dula)
        - Added cyclical temporal features (hour-of-day, day-of-year as sin/cos pairs)
        - Fit StandardScaler on training data only for proper train/test separation
        - Built sliding-window sequence generator per (station, lead_time) group with 24-hour lookback
        - Temporal train/val/test split with validation from last 15% of training period
        - Sequences at train/test boundary draw lookback context from training data
    - Built LSTM model architecture in RunoffLSTM.py (Phase 2) (Josh Dula)
        - 2-layer stacked LSTM (64 hidden units, 0.2 dropout) with MLP head (64->32->1)
        - 54,081 trainable parameters, PyTorch Dataset and DataLoader setup
        - 7 input features: nwm_forecast, residual, lead_time_hrs, hour_sin, hour_cos, doy_sin, doy_cos
    - Sequence shapes verified: 371K train / 68K val / 176K test sequences of shape (24, 7)
    - Created new repo to seprate the midterm and final material (Carson Fouts)
    - Updated preprocessing to check for any corrupt data (Carson Fouts)
    - Added train/test split for the hurricane damage dataset (Carson Fouts)
    - Reviewing Runoff data work (Quinn Lautenslager)

Tasks In Progress (4/19/2026 - 4/26/2026):
    - Adding Pytorch CNN for hurricane damage (Carson Fouts)
    - Compare LSTM-corrected forecasts vs raw NWM vs observed USGS

Week of 4/20/2026 - 4/26/2026
Authors:
    - Josh Dula
    - Carson Fouts
    - Quinn Lautenslager

Tasks Completed:
    - Built LSTM training loop in RunoffLSTM.py (Phase 3) (Josh Dula, Quinn Lautenslager)
        - Adam optimizer + MSELoss on scaled residuals (Han & Morrison 2022; Kratzert et al. 2018)
        - ReduceLROnPlateau scheduler (factor=0.5, patience=3) tied to val loss
        - Gradient clipping (max_norm=1.0) for stacked-LSTM stability (Kratzert et al. 2018 §3.2)
        - Early stopping (patience=10), best weights saved to lstm_runoff.pt
        - History dict of train/val loss + LR per epoch
        - In a separate branch tried another implementation to see if there could be any improvements
        - Graphed training history with training vs. validation plotted
    - Built test-set evaluation in RunoffLSTM.py (Phase 4) (Josh Dula)
        - Reload best checkpoint, predict residuals, inverse-transform to cfs
        - Corrected forecast = raw_nwm - predicted_residual
        - RMSE, MAE, NSE reported overall, per-station, and per-station × per-lead-time
        - Metrics written to pics/metrics_overall.csv and pics/metrics_per_lead.csv
    - Added visualizations (Phase 4) (Josh Dula)
        - pics/loss_curves.png — train vs val loss
        - pics/rmse_by_lead.png — RMSE vs lead time per station, corrected vs raw NWM
        - pics/timeseries.png — peak-flow 14-day window per station, leads {1,6,12,18}h
        - pics/scatter.png — predicted vs observed, corrected and raw NWM
    - Trained the model end-to-end on the full dataset (Josh Dula)
        - 16 epochs before early stopping, best val loss around epoch 6, ~8 min on CPU
        - Fixed a display bug where lead times were showing up as z-scores instead of 1-18h
    - Noticed the results looked too good and think there is a data leak (Josh Dula)
        - RMSE went from 56 cfs down to 1.7 cfs, and corrected error stayed almost flat
          across all 18 lead times, which shouldn't happen — longer leads should be harder
        - Looks like `residual` being in the feature list lets the model see the observed
          streamflow from the hour before the prediction time, even for long-lead forecasts
          where that observation wouldn't actually be available yet
        - Deleted the saved model and plots so we don't accidentally use them in the writeup
        - In secondary branch tried to find and fix data leak but got similar results
   - Built HurricaneDamageDataset.py: custom PyTorch Dataset loading MASK images with Label_1 as ground truth (Carson Fouts)
   - Built HurricaneDamageCNN.py: YOLO nano CNN classifier implemented from scratch in PyTorch (Carson Fouts)
   - Ran HurricaneDamagePreprocessing.py to generate hurricane_train/val/test_labels.csv (Carson Fouts)

Week of 4/27/2026 - 5/3/2026
Authors:
    - Josh Dula

Tasks Completed:
    - Fixed data leak in RunoffLSTM.py (Josh Dula)
        - Removed `residual` from FEAT_COLS — at lookback position t-k it required
          obs(t-k), but for a forecast with lead L issued at t-L, observations from
          t-L+1 through t-1 are not yet available, so including them leaked
          observations from after the forecast issue time
        - Moved scaler fitting before the train/val temporal split so feat_scaler
          and tgt_scaler are fit on train-only rows (excluding val period); prior
          version fit on the full train_val.csv which leaked val statistics into
          feature normalization
        - Smoking-gun symptom (errors flat across all 18 lead times) is now gone
          — corrected RMSE grows monotonically with lead time as it physically
          should at the wet station
    - Added obs_at_issue feature (Josh Dula)
        - USGS observation at the forecast issue time (model_initialization_time);
          legitimately known when the forecast is made, so no leak
        - Updated RunoffPreprocessing.ipynb merge cell to derive obs_at_issue
          via a second join on (streamID, model_initialization_time)
        - Regenerated data/processed/{train_val,test,merged_all}.csv with the
          new column; 5,029 train rows dropped where issue time predates USGS
          coverage (test data fully covered)
        - Without this feature the leak-free model could only learn fixed bias
          corrections per (station, lead, time-of-year) and was worse than raw
          NWM at the wet station; obs_at_issue gives it the current-state signal
    - Retrained LSTM and rebuilt plots (Josh Dula)
        - 7 features (added obs_at_issue), 54,081 params, ~9 min on CPU
        - Best val MSE 0.0044 at epoch 7, early-stopped at epoch 17
        - Station 11266500 (wet): NSE 0.814 -> 0.892, RMSE 7.30 -> 5.56 cfs
          (24% reduction); correction gap widens with lead (1h: -0.13 cfs vs
          18h: +2.81 cfs RMSE improvement) — value-add largest where NWM is
          weakest, which is the right shape for forecast post-processing
        - Station 09520500 (dry): raw NWM is wildly biased at this near-zero
          gauge (RMSE grows 8 -> 97 cfs across leads); corrected RMSE stays in
          1.8 - 3.0 cfs across all 18 leads. The huge negative NSE values are
          artifacts of dividing by tiny observed variance, not bad predictions
        - Refreshed pics/loss_curves.png, rmse_by_lead.png, timeseries.png,
          scatter.png and pics/metrics_overall.csv, pics/metrics_per_lead.csv

Tasks In Progress (5/4/2026 - 5/10/2026):
    - Write up results and the lessons learned from the leak (Josh Dula)
    - Run training and evaluate on test set with confusion matrix (Carson Fouts)
    - Ensure env is working properly and add steps to using the CNN (Carson Fouts)

