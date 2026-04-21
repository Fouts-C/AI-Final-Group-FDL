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
    - Add training loop with early stopping and LR scheduling (Phase 3) (Josh Dula)
    - Add test-set evaluation with RMSE, MAE, NSE metrics (Phase 4) (Josh Dula)
    - Generate visualizations: loss curves, RMSE by lead time, time series comparison, scatter plots (Josh Dula)
    - Adding Pytorch CNN for hurricane damage (Carson Fouts)
    - Compare LSTM-corrected forecasts vs raw NWM vs observed USGS