"""Utility functions for MLB win prediction."""

import random
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from mlb_win_pred.config import Config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def setup_logger(name: str = "mlb_win_pred", level: int = logging.INFO) -> logging.Logger:
    """Set up a logger.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def get_data_path(config: Config, filename: str, subdir: str = "raw") -> Path:
    """Get path to a data file.
    
    Args:
        config: Configuration object
        filename: Name of the file
        subdir: Subdirectory (raw or processed)
        
    Returns:
        Full path to the file
    """
    if subdir == "raw":
        return config.data_raw_dir / filename
    elif subdir == "processed":
        return config.data_processed_dir / filename
    else:
        raise ValueError(f"Unknown subdir: {subdir}")


def get_model_path(config: Config, model_name: str) -> Path:
    """Get path to a model file.
    
    Args:
        config: Configuration object
        model_name: Name of the model file
        
    Returns:
        Full path to the model file
    """
    return config.models_dir / model_name


def get_report_path(config: Config, filename: str) -> Path:
    """Get path to a report file.
    
    Args:
        config: Configuration object
        filename: Name of the report file
        
    Returns:
        Full path to the report file
    """
    return config.reports_dir / filename


def get_figure_path(config: Config, filename: str) -> Path:
    """Get path to a figure file.
    
    Args:
        config: Configuration object
        filename: Name of the figure file
        
    Returns:
        Full path to the figure file
    """
    return config.figures_dir / filename


def get_team_logo_path(team_abbr: str) -> Optional[Path]:
    """Get path to team logo file.
    
    Args:
        team_abbr: Team abbreviation (e.g., "LAD", "SFG")
        
    Returns:
        Path to logo file if it exists, None otherwise
    """
    # Logo files are expected in assets/logos/ relative to project root
    # Path is relative to project root (where config.py determines it)
    project_root = Path(__file__).parent.parent.parent
    logo_path = project_root / "assets" / "logos" / f"{team_abbr}.png"
    
    if logo_path.exists():
        return logo_path
    return None
