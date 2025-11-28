# MLB Win Probability Prediction

A machine learning project for predicting MLB game outcomes using pre-game statistics. This project uses data from the 2022-2024 MLB regular seasons to train models that predict the probability that a given team will win before the first pitch.

## Project Overview

This project implements a complete end-to-end machine learning pipeline for MLB win prediction:

- **Data Collection**: Downloads MLB data using `pybaseball` (seasons 2022-2024)
- **Feature Engineering**: Creates season-to-date statistics, rolling averages, and context features
- **Model Training**: Trains both baseline (logistic regression) and advanced (XGBoost) models
- **Evaluation**: Comprehensive evaluation with metrics and visualizations
- **Prediction CLI**: Simple command-line interface for making predictions

## Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   └── mlb_win_pred/
│       ├── __init__.py
│       ├── config.py          # Configuration settings
│       ├── data_download.py    # Download MLB data via pybaseball
│       ├── dataset_builder.py  # Build game-level dataset
│       ├── feature_engineering.py  # Feature creation
│       ├── train.py            # Model training
│       ├── evaluate.py         # Model evaluation
│       ├── predict.py          # Prediction CLI
│       ├── utils.py            # Utility functions
│       ├── theme.py            # MLB-themed UI colors and styling
│       └── predictions_archive.py  # Prediction history tracking
├── dashboard/
│   ├── dash_app.py            # Plotly Dash dashboard
│   └── streamlit_app.py       # Streamlit dashboard
├── assets/
│   └── logos/                  # Team logo images (user-provided)
├── data/
│   ├── raw/                    # Raw downloaded data
│   └── processed/              # Processed datasets (includes predictions_history.csv)
├── models/                     # Trained models
├── reports/
│   ├── figures/                # Evaluation plots
│   └── metrics.json           # Evaluation metrics
└── notebooks/
    ├── 01_exploration.ipynb    # Data exploration
    └── 02_model_debug.ipynb    # Model analysis
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Download Data

Download MLB data for the specified seasons (default: 2022-2024):

```bash
python -m mlb_win_pred.data_download
```

This will download:
- Schedule and record data for each team
- Team batting statistics
- Team pitching statistics

Data is saved to `data/raw/` and will not be re-downloaded unless `force=True` is set.

### Step 2: Build Dataset

Process raw data and create features:

```bash
python -m mlb_win_pred.dataset_builder
```

This creates a game-level dataset with:
- One row per team-game
- Season-to-date statistics (no data leakage)
- Rolling averages (last 5 and 10 games)
- Context features (home/away, month, days rest)

The processed dataset is saved to `data/processed/games_processed.csv`.

### Step 3: Train Models

Train the models:

```bash
python -m mlb_win_pred.train
```

This will:
- Split data into train/val/test sets based on dates
- Train a baseline logistic regression model
- Train an XGBoost model
- Select the best model based on validation ROC-AUC
- Retrain the best model on train+val and save to `models/`

### Step 4: Evaluate Models

Evaluate the trained model:

```bash
python -m mlb_win_pred.evaluate
```

This generates:
- Evaluation metrics (accuracy, ROC-AUC, log loss, Brier score)
- ROC curve plot
- Calibration curve plot
- Feature importance plot (for XGBoost)
- Probability distribution plot
- Metrics saved to `reports/metrics.json`

### Step 5: Make Predictions

Use the CLI to predict win probability for a specific game:

```bash
python -m mlb_win_pred.predict \
    --team "LAD" \
    --opponent "SFG" \
    --game_date "2024-06-15"
```

**Note**: The prediction CLI currently requires that the game exists in the processed dataset. For fully dynamic predictions on future games, you would need to extend the pipeline to fetch current season statistics.

All predictions made via CLI, Dash, or Streamlit are automatically saved to `data/processed/predictions_history.csv` for tracking and analysis.

### Step 6: Launch Dashboards

#### Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

#### Dash Dashboard

```bash
python dashboard/dash_app.py
```

Both dashboards feature:
- **MLB-themed UI**: Consistent color scheme with deep navy, rich red, and gold accents
- **Team logos**: Display team logos (place logo files in `assets/logos/` named by team abbreviation, e.g., `LAD.png`, `SFG.png`)
- **Prediction history**: View and filter all past predictions
- **Interactive charts**: Feature importance, historical performance, and probability gauges

## Data Splits

The project uses date-based splits to avoid temporal leakage:

- **Train**: Games before 2024-01-01 (2022-2023 seasons)
- **Validation**: Games from 2024-01-01 to 2024-07-01
- **Test**: Games from 2024-07-01 onwards

These dates can be adjusted in `config.py`.

## Features

The model uses the following feature categories:

### Team Features (with `team_` prefix)
- Season-to-date batting stats: batting average, OBP, SLG, OPS, runs per game
- Season-to-date pitching stats: ERA, WHIP, HR/9, K/9, BB/9
- Rolling stats: runs scored/allowed/differential (last 5 and 10 games)
- Days of rest

### Opponent Features (with `opp_` prefix)
- Same features as team features, but for the opponent

### Context Features
- `is_home`: Binary indicator (1 if home, 0 if away)
- `month`: Month of the game (1-12)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)

## Model Details

### Baseline Model
- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler
- **Purpose**: Simple baseline for comparison

### Main Model
- **Algorithm**: XGBoost Classifier
- **Hyperparameters** (defaults in `config.py`):
  - `n_estimators`: 200
  - `learning_rate`: 0.1
  - `max_depth`: 6
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
- **Early stopping**: 20 rounds on validation set

## Notebooks

### 01_exploration.ipynb
Exploratory data analysis including:
- Overall win rates
- Home vs away performance
- Feature correlations
- Distribution analysis
- Time series trends
- Team performance rankings

### 02_model_debug.ipynb
Model analysis including:
- Feature importances
- Calibration analysis by deciles
- Case studies (high confidence predictions)
- Prediction distributions

## Limitations and Future Improvements

### Current Limitations
1. **Data Availability**: Predictions require games to exist in the processed dataset. For future games, you'd need to extend the pipeline to fetch current season stats.
2. **Feature Engineering**: Some features may have missing values filled with medians, which could be improved.
3. **Model Selection**: Currently uses default XGBoost hyperparameters. Hyperparameter tuning could improve performance.
4. **Data Source**: Relies on `pybaseball` which may have rate limits or data format changes.

### Future Improvements
1. **Hyperparameter Tuning**: Use Optuna or similar for automated hyperparameter optimization
2. **Advanced Features**: Add pitcher matchups, weather data, rest days, travel distance
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Real-time Predictions**: Build API for making predictions on future games
5. **Model Monitoring**: Track model performance over time and detect drift
6. **Feature Store**: Implement a feature store for easier feature management
7. **Deep Learning**: Experiment with neural networks for non-linear patterns

## Features

### MLB-Themed UI

Both dashboards use a consistent MLB-themed color palette:
- **Primary**: Deep navy blue (#0C2340)
- **Secondary**: Rich red (#C8102E)
- **Accent**: Gold (#FFC72C)
- Applied to charts, buttons, headers, and layout elements

The theme is defined in `src/mlb_win_pred/theme.py` and can be customized.

### Team Logos

The dashboards support displaying team logos:
1. Create `assets/logos/` directory in the project root
2. Place logo images named by team abbreviation (e.g., `LAD.png`, `SFG.png`, `NYY.png`)
3. Logos will be displayed automatically when available
4. If a logo is missing, the team abbreviation will be shown instead

**Note for Dash**: Dash serves static assets from `dashboard/assets/`, so you may need to create a symlink or copy logos to `dashboard/assets/logos/` for Dash to serve them.

### Predictions Archive

All predictions are automatically saved to `data/processed/predictions_history.csv` with the following information:
- Timestamp (UTC)
- Team and opponent
- Game date
- Predicted win probability
- Actual result (if known, 1 for win, 0 for loss, None if unknown)
- Source (cli, dash, or streamlit)

**Viewing History**:
- **Dash**: Use the "Prediction History" tab with team filtering
- **Streamlit**: Scroll to the "Prediction History" section at the bottom
- **CLI**: Predictions are logged automatically

**Resetting Archive**: Simply delete `data/processed/predictions_history.csv` to start fresh. A new file will be created on the next prediction.

## Configuration

Edit `src/mlb_win_pred/config.py` to customize:
- Seasons to use
- Date splits for train/val/test
- XGBoost hyperparameters
- Random seed
- File paths

## Dependencies

Key dependencies:
- `pybaseball`: MLB data download
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning utilities
- `xgboost`: Gradient boosting model
- `matplotlib` & `seaborn`: Visualization
- `joblib`: Model serialization

See `requirements.txt` for full list.

## License

This project is for educational and research purposes. MLB data is used in accordance with pybaseball's terms.

## Contributing

Feel free to submit issues or pull requests for improvements!

## Contact

For questions or issues, please open an issue on the project repository.

