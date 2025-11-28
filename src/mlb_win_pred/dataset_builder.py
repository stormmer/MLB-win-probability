"""Build game-level dataset from raw MLB data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from mlb_win_pred.config import Config
from mlb_win_pred.utils import get_data_path, setup_logger
from mlb_win_pred.feature_engineering import add_features

logger = setup_logger()


def load_raw_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data files.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (schedule_df, batting_df, pitching_df)
    """
    logger.info("Loading raw data...")
    
    all_schedules = []
    all_batting = []
    all_pitching = []
    
    for season in config.seasons:
        # Load schedule data
        schedule_file = get_data_path(config, f"schedule_{season}.csv", subdir="raw")
        if schedule_file.exists():
            df = pd.read_csv(schedule_file)
            all_schedules.append(df)
        else:
            logger.warning(f"Schedule file not found: {schedule_file}")
        
        # Load team batting stats
        batting_file = get_data_path(config, f"team_batting_{season}.csv", subdir="raw")
        if batting_file.exists():
            df = pd.read_csv(batting_file)
            all_batting.append(df)
        
        # Load team pitching stats
        pitching_file = get_data_path(config, f"team_pitching_{season}.csv", subdir="raw")
        if pitching_file.exists():
            df = pd.read_csv(pitching_file)
            all_pitching.append(df)
    
    schedule_df = pd.concat(all_schedules, ignore_index=True) if all_schedules else pd.DataFrame()
    batting_df = pd.concat(all_batting, ignore_index=True) if all_batting else pd.DataFrame()
    pitching_df = pd.concat(all_pitching, ignore_index=True) if all_pitching else pd.DataFrame()
    
    logger.info(f"Loaded {len(schedule_df)} schedule records")
    logger.info(f"Loaded {len(batting_df)} batting records")
    logger.info(f"Loaded {len(pitching_df)} pitching records")
    
    return schedule_df, batting_df, pitching_df


def build_game_level_dataset(
    schedule_df: pd.DataFrame,
    batting_df: pd.DataFrame,
    pitching_df: pd.DataFrame
) -> pd.DataFrame:
    """Build game-level dataset from schedule and stats.
    
    Args:
        schedule_df: Schedule data from pybaseball
        batting_df: Team batting stats
        pitching_df: Team pitching stats
        
    Returns:
        DataFrame with one row per team-game
    """
    logger.info("Building game-level dataset...")
    
    if schedule_df.empty:
        logger.error("Schedule dataframe is empty!")
        return pd.DataFrame()
    
    # Standardize date column name
    date_cols = ['Date', 'date', 'game_date', 'Gm#']
    date_col = None
    for col in date_cols:
        if col in schedule_df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.error("Could not find date column in schedule data")
        return pd.DataFrame()
    
    schedule_df = schedule_df.rename(columns={date_col: 'game_date'})
    schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'], errors='coerce')
    
    # Filter out invalid dates
    schedule_df = schedule_df.dropna(subset=['game_date'])
    
    # Filter out postponed/suspended games (common indicators)
    if 'Tm' in schedule_df.columns:
        schedule_df = schedule_df[~schedule_df['Tm'].str.contains('postponed|suspended|cancelled', case=False, na=False)]
    
    # Extract team and opponent
    # pybaseball schedule format varies, try common column names
    team_cols = ['team', 'Tm', 'Team']
    team_col = None
    for col in team_cols:
        if col in schedule_df.columns:
            team_col = col
            break
    
    if team_col is None:
        logger.error("Could not find team column in schedule data")
        return pd.DataFrame()
    
    schedule_df = schedule_df.rename(columns={team_col: 'team'})
    
    # Extract opponent (common column names)
    opp_cols = ['Opp', 'Opponent', 'opponent']
    opp_col = None
    for col in opp_cols:
        if col in schedule_df.columns:
            opp_col = col
            break
    
    if opp_col:
        schedule_df = schedule_df.rename(columns={opp_col: 'opponent'})
    else:
        logger.warning("Could not find opponent column, will try to infer")
    
    # Extract runs scored and allowed
    # Common column names in pybaseball schedule
    runs_scored_cols = ['R', 'Runs', 'RS', 'runs_scored']
    runs_allowed_cols = ['RA', 'Runs_Allowed', 'runs_allowed']
    
    runs_scored_col = None
    for col in runs_scored_cols:
        if col in schedule_df.columns:
            runs_scored_col = col
            break
    
    runs_allowed_col = None
    for col in runs_allowed_cols:
        if col in schedule_df.columns:
            runs_allowed_col = col
            break
    
    # If we have W/L column, we can infer runs
    if 'W/L' in schedule_df.columns:
        # Split W/L column to get score if available
        # Format might be "W 5-3" or similar
        pass
    
    # Build base game dataset
    games = []
    
    for idx, row in schedule_df.iterrows():
        team = row['team']
        game_date = row['game_date']
        
        # Try to get opponent
        if 'opponent' in row and pd.notna(row['opponent']):
            opponent = row['opponent']
        else:
            # Skip if we can't determine opponent
            continue
        
        # Get runs scored and allowed
        runs_scored = None
        runs_allowed = None
        
        if runs_scored_col and pd.notna(row.get(runs_scored_col)):
            try:
                runs_scored = int(row[runs_scored_col])
            except (ValueError, TypeError):
                pass
        
        if runs_allowed_col and pd.notna(row.get(runs_allowed_col)):
            try:
                runs_allowed = int(row[runs_allowed_col])
            except (ValueError, TypeError):
                pass
        
        # If we don't have runs, try to infer from W/L or score
        if runs_scored is None or runs_allowed is None:
            if 'W/L' in row and pd.notna(row['W/L']):
                wl = str(row['W/L']).strip()
                # Try to parse score from W/L column if it contains score
                # This is format-dependent, so we'll skip for now
                pass
        
        # Determine home/away
        is_home = 1
        if '@' in str(opponent) or 'vs' in str(opponent).lower():
            # Common indicators in schedule data
            if '@' in str(opponent):
                is_home = 0
            else:
                is_home = 1
        
        # Clean opponent name (remove @ or vs)
        opponent_clean = str(opponent).replace('@', '').replace('vs', '').replace('VS', '').strip()
        
        if runs_scored is not None and runs_allowed is not None:
            win = 1 if runs_scored > runs_allowed else 0
            
            games.append({
                'team': team,
                'opponent': opponent_clean,
                'game_date': game_date,
                'runs_scored': runs_scored,
                'runs_allowed': runs_allowed,
                'is_home': is_home,
                'win': win
            })
    
    if not games:
        logger.error("No valid games found in schedule data")
        return pd.DataFrame()
    
    games_df = pd.DataFrame(games)
    
    # Handle double-headers by adding game number
    games_df = games_df.sort_values(['team', 'game_date'])
    games_df['game_num'] = games_df.groupby(['team', 'game_date']).cumcount() + 1
    
    # Merge with batting and pitching stats if available
    # Note: This is a simplified merge - in practice you'd want to match by team and season
    if not batting_df.empty:
        # Try to merge batting stats by team and season
        batting_df['season'] = pd.to_datetime(batting_df.get('Date', batting_df.get('date', ''))).dt.year if 'Date' in batting_df.columns or 'date' in batting_df.columns else batting_df.get('season', games_df['game_date'].dt.year.iloc[0])
        games_df['season'] = games_df['game_date'].dt.year
        
        # Standardize team name column in batting_df
        team_col_batting = None
        for col in ['Tm', 'Team', 'team']:
            if col in batting_df.columns:
                team_col_batting = col
                break
        
        if team_col_batting:
            batting_merge = batting_df[[team_col_batting, 'season'] + [c for c in batting_df.columns if c not in [team_col_batting, 'season']]].copy()
            batting_merge = batting_merge.rename(columns={team_col_batting: 'team'})
            # For now, we'll use these stats in feature engineering
            # Store them for later use
            games_df = games_df.merge(
                batting_merge[['team', 'season']],
                on=['team', 'season'],
                how='left'
            )
    
    logger.info(f"Built {len(games_df)} game records")
    
    return games_df


def train_val_test_split(
    df: pd.DataFrame,
    config: Config,
    date_col: str = "game_date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/val/test based on dates.
    
    Args:
        df: Full dataset
        config: Configuration object
        date_col: Column name for game date
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_end = pd.to_datetime(config.train_end_date)
    val_end = pd.to_datetime(config.val_end_date)
    test_start = pd.to_datetime(config.test_start_date)
    
    train_df = df[df[date_col] < train_end].copy()
    val_df = df[(df[date_col] >= train_end) & (df[date_col] < val_end)].copy()
    test_df = df[df[date_col] >= test_start].copy()
    
    logger.info(f"Train set: {len(train_df)} games ({train_df[date_col].min()} to {train_df[date_col].max()})")
    logger.info(f"Val set: {len(val_df)} games ({val_df[date_col].min() if not val_df.empty else 'N/A'} to {val_df[date_col].max() if not val_df.empty else 'N/A'})")
    logger.info(f"Test set: {len(test_df)} games ({test_df[date_col].min() if not test_df.empty else 'N/A'} to {test_df[date_col].max() if not test_df.empty else 'N/A'})")
    
    return train_df, val_df, test_df


def build_full_dataset(config: Config, save: bool = True) -> pd.DataFrame:
    """Build full dataset with features and save to disk.
    
    Args:
        config: Configuration object
        save: Whether to save the processed dataset
        
    Returns:
        Full dataset with features
    """
    logger.info("Building full dataset...")
    
    # Load raw data
    schedule_df, batting_df, pitching_df = load_raw_data(config)
    
    if schedule_df.empty:
        logger.error("No schedule data available. Please run data_download.py first.")
        return pd.DataFrame()
    
    # Build game-level dataset
    games_df = build_game_level_dataset(schedule_df, batting_df, pitching_df)
    
    if games_df.empty:
        logger.error("Failed to build game-level dataset")
        return pd.DataFrame()
    
    # Add features
    games_df = add_features(games_df)
    
    # Save processed dataset
    if save:
        output_file = get_data_path(config, "games_processed.csv", subdir="processed")
        games_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed dataset to {output_file}")
    
    return games_df


if __name__ == "__main__":
    from mlb_win_pred.config import get_config
    
    config = get_config()
    df = build_full_dataset(config, save=True)

