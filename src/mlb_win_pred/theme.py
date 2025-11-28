"""MLB-themed UI colors and styling for dashboards."""

from typing import Dict, Any
from dataclasses import dataclass


# MLB official color palette (matching mlb.com)
COLORS = {
    "PRIMARY": "#132448",      # MLB official navy blue (from mlb.com header)
    "SECONDARY": "#C8102E",    # MLB official red
    "ACCENT": "#132448",       # Navy for accents
    "NAVY": "#132448",         # Primary navy
    "RED": "#C8102E",          # MLB red
    "BLUE_LINK": "#0066CC",    # Link blue (for active navigation)
    "BACKGROUND_LIGHT": "#FFFFFF",  # White background (like mlb.com)
    "BACKGROUND_DARK": "#1A1A1A",   # Dark background
    "BACKGROUND_GRAY": "#F8F9FA",    # Light gray for sections
    "TEXT_PRIMARY": "#212529",       # Dark text
    "TEXT_MUTED": "#6C757D",         # Muted gray text
    "TEXT_WHITE": "#FFFFFF",         # White text
    "SUCCESS": "#28A745",             # Green for success states
    "WARNING": "#FFC107",             # Yellow for warnings
    "DANGER": "#DC3545",              # Red for errors
    "INFO": "#0066CC",                # Info blue (matching link color)
    "WHITE": "#FFFFFF",
    "LIGHT_GRAY": "#E9ECEF",
    "MEDIUM_GRAY": "#ADB5BD",
    "BORDER": "#DEE2E6",              # Light border color
}


# Typography matching MLB.com style
TYPOGRAPHY = {
    "FONT_FAMILY": "'Helvetica Neue', Helvetica, Arial, sans-serif",  # MLB-style sans-serif
    "HEADING_FONT": "'Helvetica Neue', Helvetica, Arial, sans-serif",  # For headings
    "BODY_FONT": "'Helvetica Neue', Helvetica, Arial, sans-serif",    # For body text
    "FONT_WEIGHT_BOLD": "700",
    "FONT_WEIGHT_NORMAL": "400",
}


def get_dash_theme() -> Dict[str, Any]:
    """Get theme configuration for Dash app.
    
    Returns:
        Dictionary with colors and layout defaults for Dash
    """
    return {
        "colors": {
            "primary": COLORS["PRIMARY"],
            "secondary": COLORS["SECONDARY"],
            "accent": COLORS["ACCENT"],
            "background": COLORS["BACKGROUND_LIGHT"],
            "text": COLORS["TEXT_PRIMARY"],
            "muted": COLORS["TEXT_MUTED"],
        },
        "layout": {
            "font": {
                "family": TYPOGRAPHY["FONT_FAMILY"],
                "size": 14,
                "color": COLORS["TEXT_PRIMARY"],
            },
            "paper_bgcolor": COLORS["WHITE"],
            "plot_bgcolor": COLORS["WHITE"],
            "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
        },
        "plotly_template": {
            "layout": {
                "font": {
                    "family": TYPOGRAPHY["FONT_FAMILY"],
                    "color": COLORS["TEXT_PRIMARY"],
                },
                "paper_bgcolor": COLORS["WHITE"],
                "plot_bgcolor": COLORS["WHITE"],
                "colorway": [COLORS["PRIMARY"], COLORS["SECONDARY"], COLORS["ACCENT"]],
            }
        },
    }


def get_streamlit_theme() -> Dict[str, Any]:
    """Get theme configuration suggestions for Streamlit app.
    
    Returns:
        Dictionary with color suggestions and style recommendations
    """
    return {
        "colors": {
            "primary": COLORS["PRIMARY"],
            "secondary": COLORS["SECONDARY"],
            "accent": COLORS["ACCENT"],
            "background": COLORS["BACKGROUND_LIGHT"],
            "text": COLORS["TEXT_PRIMARY"],
            "muted": COLORS["TEXT_MUTED"],
            "success": COLORS["SUCCESS"],
            "warning": COLORS["WARNING"],
            "danger": COLORS["DANGER"],
            "info": COLORS["INFO"],
        },
        "style": {
            "font_family": TYPOGRAPHY["FONT_FAMILY"],
            "heading_color": COLORS["PRIMARY"],
            "accent_color": COLORS["ACCENT"],
        },
    }


def get_plotly_template() -> Dict[str, Any]:
    """Get Plotly template configuration using MLB theme.
    
    Returns:
        Dictionary for Plotly figure.update_layout()
    """
    theme = get_dash_theme()
    return theme["plotly_template"]["layout"]

