"""Configuration settings for MLB win prediction project."""

from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    """Configuration class for MLB win prediction."""
    
    # Seasons to use
    seasons: List[int] = None
    
    # Paths
    project_root: Path = None
    data_raw_dir: Path = None
    data_processed_dir: Path = None
    models_dir: Path = None
    reports_dir: Path = None
    figures_dir: Path = None
    
    # Model settings
    random_seed: int = 42
    
    # Date splits (YYYY-MM-DD)
    train_end_date: str = "2024-01-01"
    val_end_date: str = "2024-07-01"
    test_start_date: str = "2024-07-01"
    
    # XGBoost defaults
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.seasons is None:
            self.seasons = [2022, 2023, 2024]
        
        if self.project_root is None:
            # Assume config.py is in src/mlb_win_pred, so go up 2 levels
            self.project_root = Path(__file__).parent.parent.parent
        
        if self.data_raw_dir is None:
            self.data_raw_dir = self.project_root / "data" / "raw"
        
        if self.data_processed_dir is None:
            self.data_processed_dir = self.project_root / "data" / "processed"
        
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        
        if self.reports_dir is None:
            self.reports_dir = self.project_root / "reports"
        
        if self.figures_dir is None:
            self.figures_dir = self.reports_dir / "figures"
        
        # Create directories if they don't exist
        for dir_path in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get the default configuration."""
    return Config()

