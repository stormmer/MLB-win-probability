"""Predict win probability for MLB games."""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from typing import Optional, Dict
import logging
from mlb_win_pred.config import Config
from mlb_win_pred.utils import (
    get_data_path, get_model_path, setup_logger
)
from mlb_win_pred.predictions_archive import append_prediction_record

logger = setup_logger()


def load_model(config: Config, model_path: Optional[Path] = None) -> Dict:
    """Load trained model.
    
    Args:
        config: Configuration object
        model_path: Path to model file (defaults to win_model_xgb.pkl)
        
    Returns:
        Dictionary with model, scaler, feature_cols, and model_type
    """
    if model_path is None:
        # Try XGBoost first, then logistic regression
        xgb_path = get_model_path(config, "win_model_xgb.pkl")
        lr_path = get_model_path(config, "win_model_lr.pkl")
        
        if xgb_path.exists():
            model_path = xgb_path
        elif lr_path.exists():
            model_path = lr_path
        else:
            raise FileNotFoundError(f"No model found. Expected {xgb_path} or {lr_path}")
    
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    return model_data


def predict_win_proba(
    team: str,
    opponent: str,
    game_date: str,
    config: Config,
    model_path: Optional[Path] = None
) -> float:
    """Predict win probability for a given matchup.
    
    Args:
        team: Team abbreviation (e.g., "LAD")
        opponent: Opponent abbreviation (e.g., "SFG")
        game_date: Game date in YYYY-MM-DD format
        config: Configuration object
        model_path: Path to model file
        
    Returns:
        Win probability (0-1)
    """
    # Load model
    model_data = load_model(config, model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_cols = model_data['feature_cols']
    
    # Load processed dataset to find the game
    processed_file = get_data_path(config, "games_processed.csv", subdir="processed")
    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_file}. "
            "Please run dataset_builder.py first."
        )
    
    df = pd.read_csv(processed_file)
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Find the game
    game_date_pd = pd.to_datetime(game_date)
    game = df[
        (df['team'] == team) &
        (df['opponent'] == opponent) &
        (df['game_date'] == game_date_pd)
    ]
    
    if game.empty:
        # Try to find similar matchups
        similar = df[
            ((df['team'] == team) & (df['opponent'] == opponent)) |
            ((df['team'] == opponent) & (df['opponent'] == team))
        ]
        
        if similar.empty:
            raise ValueError(
                f"No game found for {team} vs {opponent} on {game_date}. "
                "Please ensure the game exists in the processed dataset."
            )
        else:
            # Use the most recent similar game
            similar = similar.sort_values('game_date', ascending=False)
            logger.warning(
                f"Exact match not found. Using most recent similar game: "
                f"{similar.iloc[0]['game_date']}"
            )
            game = similar.iloc[0:1]
    
    # Extract features
    X = game[feature_cols].fillna(0).values
    
    # Make prediction
    if scaler is not None:
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0, 1]
    else:
        proba = model.predict_proba(X)[0, 1]
    
    return float(proba)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Predict MLB win probability for a game"
    )
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team abbreviation (e.g., LAD, NYY, BOS)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        help="Opponent abbreviation (e.g., SFG, HOU, TOR)"
    )
    parser.add_argument(
        "--game_date",
        type=str,
        required=True,
        help="Game date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file (optional, defaults to win_model_xgb.pkl)"
    )
    
    args = parser.parse_args()
    
    config = Config()
    
    try:
        proba = predict_win_proba(
            team=args.team,
            opponent=args.opponent,
            game_date=args.game_date,
            config=config,
            model_path=Path(args.model_path) if args.model_path else None
        )
        
        print(f"\nWin Probability Prediction")
        print(f"{'='*50}")
        print(f"Team: {args.team}")
        print(f"Opponent: {args.opponent}")
        print(f"Game Date: {args.game_date}")
        print(f"Win Probability: {proba:.3f} ({proba*100:.1f}%)")
        print(f"{'='*50}\n")
        
        # Append to archive
        try:
            append_prediction_record(
                team=args.team,
                opponent=args.opponent,
                game_date=args.game_date,
                predicted_prob=proba,
                actual_result=None,
                source="cli",
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to append to archive: {e}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

