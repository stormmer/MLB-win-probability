"""Archive predictions to CSV for tracking and analysis."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from mlb_win_pred.config import Config

logger = logging.getLogger(__name__)


def get_archive_path(config: Config) -> Path:
    """Get path to predictions archive CSV.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to archive CSV file
    """
    # Store archive in data/processed/ directory
    return config.data_processed_dir / "predictions_history.csv"


def append_prediction_record(
    team: str,
    opponent: str,
    game_date: str,
    predicted_prob: float,
    actual_result: Optional[int],
    source: str,
    config: Optional[Config] = None
) -> None:
    """Append a prediction record to the archive.
    
    Args:
        team: Team abbreviation
        opponent: Opponent abbreviation
        game_date: Game date in YYYY-MM-DD format
        predicted_prob: Predicted win probability (0-1)
        actual_result: Actual result (1 for win, 0 for loss, None if unknown)
        source: Source of prediction (e.g., "cli", "dash", "streamlit")
        config: Configuration object (optional, will create default if not provided)
    """
    if config is None:
        from mlb_win_pred.config import get_config
        config = get_config()
    
    archive_path = get_archive_path(config)
    
    # Create new record
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "team": team,
        "opponent": opponent,
        "game_date": game_date,
        "predicted_prob": predicted_prob,
        "actual_result": actual_result,
        "source": source,
    }
    
    # Load existing archive or create new DataFrame
    if archive_path.exists():
        try:
            df = pd.read_csv(archive_path)
        except Exception as e:
            logger.warning(f"Error reading archive: {e}. Creating new archive.")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    # Append new record
    new_row = pd.DataFrame([record])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save to CSV
    try:
        df.to_csv(archive_path, index=False)
        logger.info(f"Appended prediction record to {archive_path}")
    except Exception as e:
        logger.error(f"Error saving prediction record: {e}")


def load_predictions_archive(config: Optional[Config] = None) -> pd.DataFrame:
    """Load predictions archive as DataFrame.
    
    Args:
        config: Configuration object (optional, will create default if not provided)
        
    Returns:
        DataFrame with prediction records, or empty DataFrame if file doesn't exist
    """
    if config is None:
        from mlb_win_pred.config import get_config
        config = get_config()
    
    archive_path = get_archive_path(config)
    
    if not archive_path.exists():
        logger.info(f"Archive file not found: {archive_path}. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(archive_path)
        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])
        return df
    except Exception as e:
        logger.error(f"Error loading archive: {e}. Returning empty DataFrame.")
        return pd.DataFrame()

