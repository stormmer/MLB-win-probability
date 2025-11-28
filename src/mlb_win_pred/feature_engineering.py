"""Feature engineering for MLB win prediction."""

import pandas as pd
import numpy as np
from typing import List
import logging
from mlb_win_pred.utils import setup_logger

logger = setup_logger()


def calculate_season_to_date_stats(
    df: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "game_date",
    runs_scored_col: str = "runs_scored",
    runs_allowed_col: str = "runs_allowed",
    at_bats_col: str = "AB",
    hits_col: str = "H",
    walks_col: str = "BB",
    strikeouts_col: str = "SO",
    home_runs_col: str = "HR",
) -> pd.DataFrame:
    """Calculate season-to-date statistics for each team up to each game.
    
    Args:
        df: DataFrame with game-level data
        team_col: Column name for team
        date_col: Column name for game date
        runs_scored_col: Column name for runs scored
        runs_allowed_col: Column name for runs allowed
        at_bats_col: Column name for at bats
        hits_col: Column name for hits
        walks_col: Column name for walks
        strikeouts_col: Column name for strikeouts
        home_runs_col: Column name for home runs
        
    Returns:
        DataFrame with season-to-date stats added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([team_col, date_col])
    
    # Group by team and season
    df['season'] = df[date_col].dt.year
    
    # Calculate cumulative stats up to (but not including) each game
    stats_cols = {
        'runs_scored': runs_scored_col,
        'runs_allowed': runs_allowed_col,
        'at_bats': at_bats_col,
        'hits': hits_col,
        'walks': walks_col,
        'strikeouts': strikeouts_col,
        'home_runs': home_runs_col,
    }
    
    # Only use columns that exist
    available_cols = {k: v for k, v in stats_cols.items() if v in df.columns}
    
    for stat_name, col_name in available_cols.items():
        # Calculate cumulative sum, then shift by 1 to exclude current game
        df[f'{stat_name}_cumsum'] = df.groupby([team_col, 'season'])[col_name].cumsum()
        df[f'{stat_name}_cumsum'] = df.groupby([team_col, 'season'])[f'{stat_name}_cumsum'].shift(1)
        df[f'{stat_name}_games'] = df.groupby([team_col, 'season']).cumcount()
        df[f'{stat_name}_games'] = df.groupby([team_col, 'season'])[f'{stat_name}_games'].shift(1)
    
    # Calculate derived stats
    if 'hits_cumsum' in df.columns and 'at_bats_cumsum' in df.columns:
        df['batting_avg'] = df['hits_cumsum'] / df['at_bats_cumsum'].replace(0, np.nan)
    
    if 'hits_cumsum' in df.columns and 'walks_cumsum' in df.columns and 'at_bats_cumsum' in df.columns:
        # OBP = (H + BB) / (AB + BB)
        df['obp'] = (df['hits_cumsum'] + df['walks_cumsum']) / (
            df['at_bats_cumsum'] + df['walks_cumsum'].replace(0, np.nan)
        )
    
    if 'hits_cumsum' in df.columns and 'home_runs_cumsum' in df.columns and 'at_bats_cumsum' in df.columns:
        # SLG approximation: (H + HR) / AB (simplified)
        df['slg'] = (df['hits_cumsum'] + df['home_runs_cumsum']) / df['at_bats_cumsum'].replace(0, np.nan)
    
    if 'obp' in df.columns and 'slg' in df.columns:
        df['ops'] = df['obp'] + df['slg']
    
    if 'runs_scored_cumsum' in df.columns and 'runs_scored_games' in df.columns:
        df['runs_per_game'] = df['runs_scored_cumsum'] / df['runs_scored_games'].replace(0, np.nan)
    
    if 'strikeouts_cumsum' in df.columns and 'at_bats_cumsum' in df.columns:
        df['k_rate'] = df['strikeouts_cumsum'] / df['at_bats_cumsum'].replace(0, np.nan)
    
    if 'walks_cumsum' in df.columns and 'at_bats_cumsum' in df.columns:
        df['bb_rate'] = df['walks_cumsum'] / df['at_bats_cumsum'].replace(0, np.nan)
    
    return df


def calculate_rolling_stats(
    df: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "game_date",
    runs_scored_col: str = "runs_scored",
    runs_allowed_col: str = "runs_allowed",
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """Calculate rolling statistics over last N games.
    
    Args:
        df: DataFrame with game-level data
        team_col: Column name for team
        date_col: Column name for game date
        runs_scored_col: Column name for runs scored
        runs_allowed_col: Column name for runs allowed
        windows: List of window sizes for rolling stats
        
    Returns:
        DataFrame with rolling stats added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([team_col, date_col])
    
    for window in windows:
        # Rolling runs scored
        df[f'runs_scored_last_{window}'] = (
            df.groupby(team_col)[runs_scored_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
        )
        
        # Rolling runs allowed
        df[f'runs_allowed_last_{window}'] = (
            df.groupby(team_col)[runs_allowed_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(0, drop=True)
        )
        
        # Rolling run differential
        df[f'run_diff_last_{window}'] = (
            df[f'runs_scored_last_{window}'] - df[f'runs_allowed_last_{window}']
        )
    
    return df


def add_context_features(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    """Add context features like home/away, month, etc.
    
    Args:
        df: DataFrame with game-level data
        date_col: Column name for game date
        
    Returns:
        DataFrame with context features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Month (1-12)
    if 'month' not in df.columns:
        df['month'] = df[date_col].dt.month
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # Ensure is_home is binary (0/1)
    if 'is_home' in df.columns:
        df['is_home'] = df['is_home'].astype(int)
    
    return df


def calculate_days_rest(df: pd.DataFrame, team_col: str = "team", date_col: str = "game_date") -> pd.DataFrame:
    """Calculate days of rest for each team.
    
    Args:
        df: DataFrame with game-level data
        team_col: Column name for team
        date_col: Column name for game date
        
    Returns:
        DataFrame with days_rest added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([team_col, date_col])
    
    # Calculate days since last game
    df['prev_game_date'] = df.groupby(team_col)[date_col].shift(1)
    df['days_rest'] = (df[date_col] - df['prev_game_date']).dt.days
    df['days_rest'] = df['days_rest'].fillna(3)  # Default to 3 days for first game of season
    df = df.drop(columns=['prev_game_date'])
    
    return df


def add_features(
    df: pd.DataFrame,
    team_col: str = "team",
    opponent_col: str = "opponent",
    date_col: str = "game_date",
    runs_scored_col: str = "runs_scored",
    runs_allowed_col: str = "runs_allowed"
) -> pd.DataFrame:
    """Add all features to the dataset.
    
    This function orchestrates all feature engineering steps.
    
    Args:
        df: DataFrame with game-level data
        team_col: Column name for team
        opponent_col: Column name for opponent
        date_col: Column name for game date
        runs_scored_col: Column name for runs scored
        runs_allowed_col: Column name for runs allowed
        
    Returns:
        DataFrame with all features added
    """
    logger.info("Starting feature engineering...")
    df = df.copy()
    
    # Ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([team_col, date_col])
    
    # Add context features first
    df = add_context_features(df, date_col=date_col)
    
    # Calculate days rest
    df = calculate_days_rest(df, team_col=team_col, date_col=date_col)
    
    # Calculate rolling stats
    df = calculate_rolling_stats(
        df,
        team_col=team_col,
        date_col=date_col,
        runs_scored_col=runs_scored_col,
        runs_allowed_col=runs_allowed_col,
        windows=[5, 10]
    )
    
    # Calculate season-to-date stats for team
    df = calculate_season_to_date_stats(
        df,
        team_col=team_col,
        date_col=date_col,
        runs_scored_col=runs_scored_col,
        runs_allowed_col=runs_allowed_col
    )
    
    # Rename team features with prefix
    feature_cols = [
        'batting_avg', 'obp', 'slg', 'ops', 'runs_per_game',
        'k_rate', 'bb_rate', 'days_rest',
        'runs_scored_last_5', 'runs_allowed_last_5', 'run_diff_last_5',
        'runs_scored_last_10', 'runs_allowed_last_10', 'run_diff_last_10'
    ]
    
    for col in feature_cols:
        if col in df.columns:
            df[f'team_{col}'] = df[col]
    
    # Calculate opponent features
    # Create opponent dataframe by swapping team and opponent
    opp_df = df.copy()
    # Temporarily swap columns to calculate opponent stats
    opp_df = opp_df.rename(columns={team_col: '_temp_team', opponent_col: team_col})
    opp_df = opp_df.rename(columns={'_temp_team': opponent_col})
    
    # Calculate features for opponent (now in team_col position)
    opp_df = calculate_season_to_date_stats(
        opp_df,
        team_col=team_col,  # This is now the opponent
        date_col=date_col,
        runs_scored_col=runs_scored_col,
        runs_allowed_col=runs_allowed_col
    )
    opp_df = calculate_rolling_stats(
        opp_df,
        team_col=team_col,  # This is now the opponent
        date_col=date_col,
        runs_scored_col=runs_scored_col,
        runs_allowed_col=runs_allowed_col,
        windows=[5, 10]
    )
    opp_df = calculate_days_rest(opp_df, team_col=team_col, date_col=date_col)
    
    # Swap back for merging
    opp_df = opp_df.rename(columns={team_col: '_temp_team', opponent_col: team_col})
    opp_df = opp_df.rename(columns={'_temp_team': opponent_col})
    
    # Merge opponent features back
    opp_feature_cols = [
        'batting_avg', 'obp', 'slg', 'ops', 'runs_per_game',
        'k_rate', 'bb_rate', 'days_rest',
        'runs_scored_last_5', 'runs_allowed_last_5', 'run_diff_last_5',
        'runs_scored_last_10', 'runs_allowed_last_10', 'run_diff_last_10'
    ]
    
    merge_cols = [date_col, team_col, opponent_col]
    for col in opp_feature_cols:
        if col in opp_df.columns:
            opp_df[f'opp_{col}'] = opp_df[col]
            merge_cols.append(f'opp_{col}')
    
    # Merge opponent features
    df = df.merge(
        opp_df[merge_cols],
        on=[date_col, team_col, opponent_col],
        how='left',
        suffixes=('', '_opp')
    )
    
    # Fill NaN values with reasonable defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith('team_') or col.startswith('opp_'):
            df[col] = df[col].fillna(df[col].median())
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    logger.info(f"Feature columns: {[c for c in df.columns if c.startswith('team_') or c.startswith('opp_')]}")
    
    return df

