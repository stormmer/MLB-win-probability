"""Download MLB data using pybaseball."""

import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
from mlb_win_pred.config import Config
from mlb_win_pred.utils import get_data_path, setup_logger

logger = setup_logger()


def download_schedule_data(
    config: Config,
    seasons: Optional[List[int]] = None,
    force: bool = False
) -> pd.DataFrame:
    """Download schedule and record data for specified seasons.
    
    Args:
        config: Configuration object
        seasons: List of seasons to download (defaults to config.seasons)
        force: If True, re-download even if files exist
        
    Returns:
        Combined DataFrame of schedule data
    """
    if seasons is None:
        seasons = config.seasons
    
    all_schedules = []
    
    for season in seasons:
        filename = f"schedule_{season}.csv"
        filepath = get_data_path(config, filename, subdir="raw")
        
        if filepath.exists() and not force:
            logger.info(f"Loading existing schedule data for {season}")
            df = pd.read_csv(filepath)
        else:
            logger.info(f"Downloading schedule data for {season}")
            try:
                import pybaseball
                # Standard MLB team abbreviations
                mlb_teams = [
                    'LAD', 'SFG', 'SDP', 'COL', 'ARI',  # NL West
                    'ATL', 'PHI', 'NYM', 'MIA', 'WSN',  # NL East
                    'MIL', 'CHC', 'STL', 'CIN', 'PIT',  # NL Central
                    'HOU', 'SEA', 'TEX', 'LAA', 'OAK',  # AL West
                    'NYY', 'BOS', 'TOR', 'TBR', 'BAL',  # AL East
                    'CLE', 'MIN', 'CHW', 'DET', 'KCR'    # AL Central
                ]
                team_schedules = []
                
                import time
                for i, team_id in enumerate(mlb_teams):
                    try:
                        # Add delay to avoid rate limiting (except for first request)
                        if i > 0:
                            time.sleep(2)  # 2 second delay between requests
                        
                        schedule = pybaseball.schedule_and_record(season, team_id)
                        if schedule is not None and not schedule.empty:
                            schedule['team'] = team_id
                            schedule['season'] = season
                            team_schedules.append(schedule)
                            logger.info(f"Successfully downloaded schedule for {team_id} in {season}")
                    except Exception as e:
                        logger.warning(f"Failed to download schedule for {team_id} in {season}: {e}")
                        # Add extra delay after errors
                        time.sleep(3)
                        continue
                
                if team_schedules:
                    df = pd.concat(team_schedules, ignore_index=True)
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved schedule data for {season} to {filepath}")
                else:
                    logger.warning(f"No schedule data downloaded for {season}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error downloading schedule data for {season}: {e}")
                continue
        
        if 'df' in locals() and not df.empty:
            all_schedules.append(df)
    
    if all_schedules:
        combined = pd.concat(all_schedules, ignore_index=True)
        logger.info(f"Total schedule records: {len(combined)}")
        return combined
    else:
        logger.warning("No schedule data available")
        return pd.DataFrame()


def download_team_stats(
    config: Config,
    seasons: Optional[List[int]] = None,
    force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download team batting and pitching stats.
    
    Args:
        config: Configuration object
        seasons: List of seasons to download (defaults to config.seasons)
        force: If True, re-download even if files exist
        
    Returns:
        Tuple of (batting_df, pitching_df)
    """
    if seasons is None:
        seasons = config.seasons
    
    all_batting = []
    all_pitching = []
    
    for season in seasons:
        batting_file = get_data_path(config, f"team_batting_{season}.csv", subdir="raw")
        pitching_file = get_data_path(config, f"team_pitching_{season}.csv", subdir="raw")
        
        if batting_file.exists() and pitching_file.exists() and not force:
            logger.info(f"Loading existing team stats for {season}")
            batting_df = pd.read_csv(batting_file)
            pitching_df = pd.read_csv(pitching_file)
        else:
            logger.info(f"Downloading team stats for {season}")
            try:
                import pybaseball
                
                # Download team batting stats
                batting_df = pybaseball.team_batting(season, qual=1)
                if batting_df is not None and not batting_df.empty:
                    batting_df['season'] = season
                    batting_df.to_csv(batting_file, index=False)
                    logger.info(f"Saved team batting stats for {season}")
                
                # Download team pitching stats
                pitching_df = pybaseball.team_pitching(season, qual=1)
                if pitching_df is not None and not pitching_df.empty:
                    pitching_df['season'] = season
                    pitching_df.to_csv(pitching_file, index=False)
                    logger.info(f"Saved team pitching stats for {season}")
                    
            except Exception as e:
                logger.error(f"Error downloading team stats for {season}: {e}")
                continue
        
        if 'batting_df' in locals() and not batting_df.empty:
            all_batting.append(batting_df)
        if 'pitching_df' in locals() and not pitching_df.empty:
            all_pitching.append(pitching_df)
    
    batting_combined = pd.concat(all_batting, ignore_index=True) if all_batting else pd.DataFrame()
    pitching_combined = pd.concat(all_pitching, ignore_index=True) if all_pitching else pd.DataFrame()
    
    logger.info(f"Total batting records: {len(batting_combined)}")
    logger.info(f"Total pitching records: {len(pitching_combined)}")
    
    return batting_combined, pitching_combined


def download_all_data(config: Config, force: bool = False) -> None:
    """Download all required data.
    
    Args:
        config: Configuration object
        force: If True, re-download even if files exist
    """
    logger.info("Starting data download...")
    
    # Download schedule data
    schedule_df = download_schedule_data(config, force=force)
    
    # Download team stats
    batting_df, pitching_df = download_team_stats(config, force=force)
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    from mlb_win_pred.config import get_config
    
    config = get_config()
    download_all_data(config, force=False)

