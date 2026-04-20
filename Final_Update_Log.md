Week of 2/16/2026 - 2/22/2026
Authors:
    - Josh Dula
    - Quinn Lautenslager
    - Carson Fouts

Tasks Completed:
    - Data packages downloaded and committed to Github (Josh Dula)
    - Added Data Wrangling ZyBook pdf for referencing (Josh Dula)
    - Data Wrangling for AV dataset (Quinn Lautenslager)
    - Data Wrangling for SP dataset (Carson Fouts)
    - Updated README.md (Josh Dula)
    - Updated Midterm Update Log (Josh Dula)

Week of 2/22/2026 - 2/29/2026
Authors:
    - Josh Dula
    - Quinn Lautenslager
    - Carson Fouts

Tasks Completed:
    - AV dataset: Set up initial data exploration, decided X and y values, added Linear Regression with cross-validation (Quinn Lautenslager)
    - SP dataset: Built ML pipeline with categorical encoding, train/test split, and cross-validation for Linear Regression (Carson Fouts)
    - Reviewed branch progress and verified work across AuctionBranch and student-performance-wrangling (Josh Dula)
    - Updated Midterm Update Log (Josh Dula)
    - Updated README.md (Josh Dula)

Tasks In Progress (2/22/2026) - 2/29/2026:
    - Finalizing Data Wrangling and Analysis for AV dataset (Quinn Lautenslager)
    - Finalizing Data Wrangling and Analysis for SP dataset (Carson Fouts)
    - Discuss potential visualizations for both Datasets (Josh Dula, Carson Fouts, Quinn Lautenslager)
    - Begin working on scripts for visualizing the datasets (Josh Dula, Carson Fouts, Quinn Lautenslager)

Week of 2/22/2026 - 3/2/2026
Authors:
    - Josh Dula
    - Quinn Lautenslager
    - Carson Fouts

Tasks Completed:
    - AV dataset: Finalized data extraction with DataWranglingAV.py, confirmed Linear Regression pipeline with cross-validation (Quinn Lautenslager)
    - AV dataset: Updated DataWrangling.ipynb with expanded wrangling and model output (Quinn Lautenslager)
    - SP dataset: Created SPGenderAnalysis.py with grouped bar chart visualizing performance distribution by gender (Josh Dula)
    - SP dataset: Added zyBooks Chapter 5 references (Sections 5.1, 5.2, 5.2.3, 5.2.5) to visualization code for documentation (Josh Dula)
    - Pulled and reviewed all teammate branch updates (AuctionBranch, SPBranch) (Josh Dula)
    - Updated Midterm Update Log (Josh Dula)

Tasks In Progress (3/2/2026 - 3/9/2026):
    - Expand visualizations for AV dataset (Quinn Lautenslager)
    - Expand visualizations for SP dataset (Carson Fouts, Josh Dula)
    - Prepare midterm presentation materials (Josh Dula, Carson Fouts, Quinn Lautenslager)

Week of 3/16/2026 - 3/23/2026
Authors:
    - Josh Dula
    - Quinn Lautenslager
    - Carson Fouts

Tasks Completed:
    - SP dataset: Switched from Linear Regression to Logistic Regression for classification of student performance (Carson Fouts)
    - SP dataset: Added Confusion Matrix visualization and Decision Boundary Plot using sklearn's DecisionBoundaryDisplay (Carson Fouts)
    - SP dataset: Created standalone DataWranglingSP.py with full ML pipeline including one-hot encoding, Logistic Regression, cross-validation, and visualizations (Carson Fouts)
    - AV dataset: Added KNN Regressor with GridSearchCV hyperparameter tuning (k = 3, 5, 7, 9, 11, 13, 15) to DataWrangling.ipynb (Quinn Lautenslager)
    - AV dataset: Built sklearn Pipeline with MinMaxScaler and KNeighborsRegressor for streamlined prediction (Quinn Lautenslager)
    - AV dataset: Added Actual vs Predicted scatter plot visualizations for KNN Regressor results (Quinn Lautenslager)
    - Pulled and reviewed all teammate branch updates (AuctionBranch, SPBranch) (Josh Dula)
    - Updated Midterm Update Log (Josh Dula)

Tasks In Progress (3/23/2026 - 3/30/2026):
    - Continue refining model performance and evaluation for both datasets (Quinn Lautenslager, Carson Fouts)
    - Prepare midterm presentation materials (Josh Dula, Carson Fouts, Quinn Lautenslager)

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

Tasks In Progress (4/19/2026 - 4/26/2026):
    - Add training loop with early stopping and LR scheduling (Phase 3) (Josh Dula)
    - Add test-set evaluation with RMSE, MAE, NSE metrics (Phase 4) (Josh Dula)
    - Generate visualizations: loss curves, RMSE by lead time, time series comparison, scatter plots (Josh Dula)
    - Compare LSTM-corrected forecasts vs raw NWM vs observed USGS