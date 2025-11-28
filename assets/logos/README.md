# Team Logos

Place team logo images in this directory to display them in the dashboards.

## File Naming

Logo files should be named using team abbreviations (e.g., `LAD.png`, `SFG.png`, `NYY.png`).

## Supported Formats

- PNG (recommended)
- JPG/JPEG

## Where to Get Logos

You can download team logos from:
- Official MLB team websites
- Wikimedia Commons
- Other public domain image sources

## Dash Dashboard

For the Dash dashboard, logos can be placed in either:
- `assets/logos/` (project root) - will be loaded via base64 encoding
- `dashboard/assets/logos/` - will be served as static assets (preferred for Dash)

## Streamlit Dashboard

For the Streamlit dashboard, logos should be in:
- `assets/logos/` (project root)

## Example Team Abbreviations

Common MLB team abbreviations:
- LAD (Los Angeles Dodgers)
- SFG (San Francisco Giants)
- NYY (New York Yankees)
- BOS (Boston Red Sox)
- HOU (Houston Astros)
- ATL (Atlanta Braves)
- ... and all other MLB teams

If a logo is not found, the dashboard will display the team abbreviation as text instead.

