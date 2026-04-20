# AI-Midterm-Group-FDL
- AI project with Carson Fouts, Josh Dula, and Quinn Lautenslager

## Final Project: Improving NWM Forecasts Using Deep Learning Post-processing
- Using a Deep Learning model (LSTM) to post-process National Water Model (NWM) short-range runoff forecasts
- Two USGS stations, lead times 1-18 hours, April 2021 - April 2023
- Train/Val: April 2021 - September 2022, Test: October 2022 - April 2023

References:
- Han, H. & Morrison, R. R. (2022). Improved runoff forecasting performance through error predictions using a deep-learning approach. Journal of Hydrology, 608, 127653.
- zyBooks Chapter 5 - Data Wrangling

## Tasks Completed

### Midterm (Weeks of 2/16 - 3/23)
- Data packages downloaded and committed to Github (Josh Dula)
- Added Data Wrangling ZyBook pdf for referencing (Josh Dula)
- Data Wrangling for AV dataset (Quinn Lautenslager)
- Data Wrangling for SP dataset (Carson Fouts)
- AV dataset: Data exploration, feature selection, Linear Regression with cross-validation (Quinn Lautenslager)
- AV dataset: KNN Regressor with GridSearchCV, RandomForest (Quinn Lautenslager)
- SP dataset: ML pipeline with categorical encoding, Logistic Regression, confusion matrix, decision boundary plots (Carson Fouts)
- Reviewed branch progress and verified work (Josh Dula)
- Updated README.md and Midterm Update Log (Josh Dula)

### Week of 4/6/2026 - 4/12/2026
- Added all new runoff forecasting datasets (NWM forecasts + USGS observations for 2 stations) (Quinn Lautenslager)
- Combined NWM streamflow CSVs and USGS observed CSVs into combined files in RunoffPreprocessing.ipynb (Quinn Lautenslager)
- Added data preprocessing pipeline to RunoffPreprocessing.ipynb (Josh Dula):
    - Fixed USGS column mismatch between the two station files
    - Parsed datetime formats for both NWM and USGS data
    - Computed lead times (1-18h) from NWM forecast timestamps
    - Resampled USGS from 15-min to hourly to match NWM
    - Cleaned missing values and duplicates
    - Merged NWM forecasts with USGS observations, derived residual feature
    - Split into train/val (Apr 2021 - Sep 2022) and test (Oct 2022 - Apr 2023)
    - Saved processed data to data/processed/

## Tasks In Progress
- Build LSTM model for runoff forecast error correction (Josh Dula, Quinn Lautenslager, Carson Fouts)
- Evaluate model performance with RMSE, NSE metrics
- Compare corrected forecasts vs original NWM vs observed USGS
