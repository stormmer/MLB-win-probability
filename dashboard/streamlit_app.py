"""Streamlit dashboard for MLB win prediction."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import joblib
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Tuple
import logging

from mlb_win_pred.config import get_config
from mlb_win_pred.utils import (
    get_data_path, get_model_path, get_report_path, setup_logger, get_team_logo_path
)
from mlb_win_pred.predict import load_model, predict_win_proba
from mlb_win_pred.theme import get_streamlit_theme, COLORS, get_plotly_template
from mlb_win_pred.predictions_archive import append_prediction_record, load_predictions_archive

logger = setup_logger()

# Page config
st.set_page_config(
    page_title="MLB Win Probability Dashboard",
    page_icon="⚾",
    layout="wide"
)

# Load configuration
config = get_config()


@st.cache_resource
def load_model_cached():
    """Load trained model with caching.
    
    Returns:
        Dictionary with model, scaler, feature_cols, and model_type
    """
    try:
        model_data = load_model(config)
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_data_cached():
    """Load processed dataset with caching.
    
    Returns:
        DataFrame with processed games
    """
    try:
        processed_file = get_data_path(config, "games_processed.csv", subdir="processed")
        if not processed_file.exists():
            st.error(f"Processed dataset not found: {processed_file}")
            return None
        
        df = pd.read_csv(processed_file)
        df['game_date'] = pd.to_datetime(df['game_date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def load_metrics():
    """Load evaluation metrics with caching.
    
    Returns:
        Dictionary of metrics or None
    """
    try:
        metrics_path = get_report_path(config, "metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.warning(f"Could not load metrics: {e}")
        return None


def get_feature_importance(model_data: Dict) -> pd.DataFrame:
    """Extract feature importance from model.
    
    Args:
        model_data: Dictionary containing model and feature columns
        
    Returns:
        DataFrame with feature names and importances
    """
    if model_data is None:
        return pd.DataFrame()
    
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return df
    else:
        return pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})


def get_historical_games(
    df: pd.DataFrame,
    team: str,
    opponent: str,
    model_data: Optional[Dict],
    n: int = 10
) -> pd.DataFrame:
    """Get last N games between two teams with predictions.
    
    Args:
        df: Processed dataset
        team: Team abbreviation
        opponent: Opponent abbreviation
        model_data: Model data dictionary (optional)
        n: Number of games to return
        
    Returns:
        DataFrame with historical games and predictions
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Get games where team played opponent (either as team or opponent)
    historical = df[
        ((df['team'] == team) & (df['opponent'] == opponent)) |
        ((df['team'] == opponent) & (df['opponent'] == team))
    ].sort_values('game_date', ascending=False).head(n)
    
    if historical.empty:
        return pd.DataFrame()
    
    # Calculate predictions if model is available
    if model_data is not None:
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_cols = model_data['feature_cols']
        
        predicted_probas = []
        for idx, row in historical.iterrows():
            try:
                # Get features for this game
                X = row[feature_cols].fillna(0).values.reshape(1, -1)
                
                # Make prediction
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                    proba = model.predict_proba(X_scaled)[0, 1]
                else:
                    proba = model.predict_proba(X)[0, 1]
                
                # If this row is opponent as team, flip probability
                if row['team'] == opponent:
                    proba = 1 - proba
                
                predicted_probas.append(proba)
            except Exception as e:
                logger.warning(f"Error predicting for game {idx}: {e}")
                predicted_probas.append(0.5)
        
        historical['predicted_proba'] = predicted_probas
    else:
        historical['predicted_proba'] = 0.5  # Default if no model
    
    historical = historical.sort_values('game_date', ascending=False)
    
    return historical


def create_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature importances
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    top_features = importance_df.head(top_n)
    theme_template = get_plotly_template()
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color=COLORS['PRIMARY'])
        )
    ])
    
    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500,
        margin=dict(l=200, r=20, t=40, b=20),
        **theme_template
    )
    
    return fig


def main():
    """Main Streamlit app."""
    # Apply MLB theme styling
    theme = get_streamlit_theme()
    
    # Custom CSS for MLB theme
    st.markdown(f"""
    <style>
        .main {{
            background-color: {COLORS['BACKGROUND_GRAY']};
            font-family: {theme['style']['font_family']};
        }}
        h1 {{
            color: {COLORS['PRIMARY']};
            font-weight: 700;
            font-family: {theme['style']['font_family']};
        }}
        h2, h3 {{
            color: {COLORS['PRIMARY']};
            font-weight: 600;
            font-family: {theme['style']['font_family']};
        }}
        .stButton>button {{
            background-color: {COLORS['PRIMARY']};
            color: {COLORS['WHITE']};
            font-weight: 600;
            border-radius: 4px;
        }}
        .stButton>button:hover {{
            background-color: {COLORS['SECONDARY']};
        }}
        .stCard {{
            border: 1px solid {COLORS['BORDER']};
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # MLB Logo and Title
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        mlb_logo_path = project_root / "assets" / "mlb_logo.png"
        if mlb_logo_path.exists():
            st.image(str(mlb_logo_path), width=80)
        else:
            st.markdown("⚾")
    with col_title:
        st.title("Win Probability Dashboard")
    
    # Load data and model
    df = load_data_cached()
    model_data = load_model_cached()
    metrics = load_metrics()
    
    if df is None or model_data is None:
        st.error("Failed to load required data or model. Please ensure the dataset and model are available.")
        return
    
    # Sidebar
    st.sidebar.header("Controls")
    
    # Get available teams
    teams = sorted(df['team'].unique().tolist())
    
    team = st.sidebar.selectbox(
        "Team:",
        options=teams,
        index=0 if teams else None
    )
    
    opponent = st.sidebar.selectbox(
        "Opponent:",
        options=teams,
        index=1 if len(teams) > 1 else None
    )
    
    # Get available dates for this matchup
    available_dates = df[
        ((df['team'] == team) & (df['opponent'] == opponent)) |
        ((df['team'] == opponent) & (df['opponent'] == team))
    ]['game_date'].dt.date.unique()
    
    if len(available_dates) > 0:
        available_dates = sorted(available_dates, reverse=True)
        selected_date = st.sidebar.selectbox(
            "Game Date:",
            options=available_dates,
            index=0
        )
        game_date_str = selected_date.strftime("%Y-%m-%d")
    else:
        st.sidebar.warning("No games found for this matchup")
        game_date_str = None
    
    # Predict button
    predict_button = st.sidebar.button(
        "Compute Win Probability",
        type="primary",
        use_container_width=True
    )
    
    # Main content
    if predict_button and game_date_str:
        try:
            # Make prediction
            proba = predict_win_proba(team, opponent, game_date_str, config)
            
            # Append to archive
            try:
                append_prediction_record(
                    team=team,
                    opponent=opponent,
                    game_date=game_date_str,
                    predicted_prob=proba,
                    actual_result=None,
                    source="streamlit",
                    config=config
                )
            except Exception as e:
                logger.warning(f"Failed to append to archive: {e}")
            
            # Display team logos
            team_logo_path = get_team_logo_path(team)
            opponent_logo_path = get_team_logo_path(opponent)
            
            logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
            with logo_col1:
                if team_logo_path and team_logo_path.exists():
                    st.image(str(team_logo_path), width=100)
                else:
                    st.markdown(f"### {team}")
            with logo_col2:
                st.markdown("### vs")
            with logo_col3:
                if opponent_logo_path and opponent_logo_path.exists():
                    st.image(str(opponent_logo_path), width=100)
                else:
                    st.markdown(f"### {opponent}")
            
            # Display probability
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### Predicted Win Probability")
                st.markdown(f"## <span style='color: {theme['colors']['accent']}'>{proba:.1%}</span>", unsafe_allow_html=True)
                
                # Progress bar
                st.progress(proba)
                
                # Interpretation
                if proba > 0.6:
                    st.success(f"Strong favorite: {team} has a {proba:.1%} chance to win")
                elif proba > 0.4:
                    st.info(f"Competitive game: {team} has a {proba:.1%} chance to win")
                else:
                    st.warning(f"Underdog: {team} has a {proba:.1%} chance to win")
            
            st.divider()
            
            # Last 10 games
            st.subheader(f"Last 10 Games: {team} vs {opponent}")
            historical = get_historical_games(df, team, opponent, model_data, n=10)
            
            if not historical.empty:
                # Prepare display dataframe
                display_df = historical[['game_date', 'team', 'opponent', 'runs_scored', 'runs_allowed', 'win']].copy()
                display_df['game_date'] = display_df['game_date'].dt.date
                
                # Determine if team won (accounting for which team is which)
                display_df['team_won'] = display_df.apply(
                    lambda row: row['win'] if row['team'] == team else (1 - row['win']),
                    axis=1
                )
                
                # Add predicted probabilities
                if 'predicted_proba' in historical.columns:
                    display_df['predicted_proba'] = historical['predicted_proba'].values
                else:
                    display_df['predicted_proba'] = 0.5
                
                display_df['result'] = display_df['team_won'].apply(lambda x: "Win" if x == 1 else "Loss")
                display_df['score'] = display_df.apply(
                    lambda row: f"{row['runs_scored']}-{row['runs_allowed']}" if row['team'] == team
                    else f"{row['runs_allowed']}-{row['runs_scored']}",
                    axis=1
                )
                
                # Display table
                display_cols = ['game_date', 'result', 'score', 'predicted_proba']
                display_df['predicted_proba'] = display_df['predicted_proba'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    display_df[display_cols].rename(columns={
                        'game_date': 'Date',
                        'result': 'Result',
                        'score': 'Score',
                        'predicted_proba': 'Predicted Prob'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No historical games found for this matchup")
            
            st.divider()
            
            # Feature importance and metrics side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance")
                importance_df = get_feature_importance(model_data)
                if not importance_df.empty:
                    fig = create_feature_importance_plot(importance_df)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")
            
            with col2:
                st.subheader("Model Performance Metrics")
                if metrics:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
                    st.metric("Log Loss", f"{metrics.get('log_loss', 0):.3f}")
                    st.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
                    
                    if 'confusion_matrix' in metrics:
                        cm = metrics['confusion_matrix']
                        st.write("**Confusion Matrix:**")
                        st.write(f"True Positives: {cm.get('tp', 0)}")
                        st.write(f"True Negatives: {cm.get('tn', 0)}")
                        st.write(f"False Positives: {cm.get('fp', 0)}")
                        st.write(f"False Negatives: {cm.get('fn', 0)}")
                else:
                    st.info("Metrics not available. Run evaluate.py to generate metrics.json")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            logger.error(f"Prediction error: {e}", exc_info=True)
    
    elif predict_button:
        st.warning("Please select a valid game date")
    else:
        # Initial state - show instructions
        st.info("👈 Select a team, opponent, and game date in the sidebar, then click 'Compute Win Probability'")
        
        # Show model info
        if model_data:
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Model Type:** {model_data.get('model_type', 'unknown')}")
            st.sidebar.markdown(f"**Features:** {len(model_data.get('feature_cols', []))}")
    
    # Prediction History Section
    st.divider()
    st.subheader("Prediction History")
    
    # Filter by team
    archive_df = load_predictions_archive(config)
    if not archive_df.empty:
        teams_in_archive = sorted(set(archive_df['team'].unique().tolist() + archive_df['opponent'].unique().tolist()))
        filter_team = st.selectbox(
            "Filter by Team:",
            options=['All Teams'] + teams_in_archive,
            index=0
        )
        
        # Filter data
        display_df = archive_df.tail(50).copy()
        if filter_team != 'All Teams':
            display_df = display_df[
                (display_df['team'] == filter_team) | 
                (display_df['opponent'] == filter_team)
            ]
        
        # Format for display
        display_df = display_df.sort_values('timestamp', ascending=False)
        display_cols = ['timestamp', 'team', 'opponent', 'game_date', 'predicted_prob', 'source']
        if all(col in display_df.columns for col in display_cols):
            display_df['predicted_prob'] = display_df['predicted_prob'].apply(lambda x: f"{x:.1%}")
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'game_date' in display_df.columns:
                display_df['game_date'] = pd.to_datetime(display_df['game_date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df[display_cols].rename(columns={
                    'timestamp': 'Timestamp',
                    'team': 'Team',
                    'opponent': 'Opponent',
                    'game_date': 'Game Date',
                    'predicted_prob': 'Predicted Prob',
                    'source': 'Source'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Archive data format is incomplete.")
    else:
        st.info("No predictions recorded yet. Make a prediction to see history here.")


if __name__ == "__main__":
    main()

