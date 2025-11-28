# MLB Win Probability Dashboards

This folder contains two interactive dashboards for the MLB win prediction project:

## Dash App (Plotly Dash)

**File:** `dash_app.py`

A modern web dashboard built with Plotly Dash.

### Features:
- Team and opponent selection dropdowns
- Date picker for game selection
- Win probability prediction with radial gauge visualization
- Feature importance bar chart
- Historical team performance chart with predictions vs actual outcomes

### Usage:
```bash
python dashboard/dash_app.py
```

Then open your browser to `http://localhost:8050`

## Streamlit App

**File:** `streamlit_app.py`

An interactive dashboard built with Streamlit.

### Features:
- Sidebar controls for team, opponent, and game selection
- Win probability display with interpretation
- Last 10 games between selected teams with predictions
- Feature importance visualization
- Model performance metrics from `reports/metrics.json`

### Usage:
```bash
streamlit run dashboard/streamlit_app.py
```

Or:
```bash
python -m streamlit run dashboard/streamlit_app.py
```

## Requirements

Both dashboards require:
- Processed dataset: `data/processed/games_processed.csv`
- Trained model: `models/win_model_xgb.pkl` or `models/win_model_lr.pkl`
- (Optional) Metrics: `reports/metrics.json` (for Streamlit app)

Make sure you've run the data pipeline and trained the model before using the dashboards:
1. `python -m mlb_win_pred.data_download`
2. `python -m mlb_win_pred.dataset_builder`
3. `python -m mlb_win_pred.train`
4. (Optional) `python -m mlb_win_pred.evaluate`

## Dependencies

Both dashboards use:
- `dash` and `plotly` for the Dash app
- `streamlit` and `plotly` for the Streamlit app

Install with:
```bash
pip install dash plotly streamlit
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

