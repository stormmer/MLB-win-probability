"""Download MLB team logos from the internet."""

import sys
from pathlib import Path
import requests
from typing import Dict, Optional
import logging

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from mlb_win_pred.utils import setup_logger

logger = setup_logger()

# MLB team abbreviations and their Wikipedia/Wikimedia image names
# Using Wikimedia Commons as a reliable source for team logos
TEAM_LOGOS = {
    "ARI": "Arizona_Diamondbacks_logo.svg",
    "ATL": "Atlanta_Braves_logo.svg",
    "BAL": "Baltimore_Orioles_logo.svg",
    "BOS": "Boston_Red_Sox_logo.svg",
    "CHC": "Chicago_Cubs_logo.svg",
    "CWS": "Chicago_White_Sox_logo.svg",
    "CIN": "Cincinnati_Reds_logo.svg",
    "CLE": "Cleveland_Guardians_logo.svg",
    "COL": "Colorado_Rockies_logo.svg",
    "DET": "Detroit_Tigers_logo.svg",
    "HOU": "Houston_Astros_logo.svg",
    "KCR": "Kansas_City_Royals_logo.svg",
    "LAA": "Los_Angeles_Angels_logo.svg",
    "LAD": "Los_Angeles_Dodgers_logo.svg",
    "MIA": "Miami_Marlins_logo.svg",
    "MIL": "Milwaukee_Brewers_logo.svg",
    "MIN": "Minnesota_Twins_logo.svg",
    "NYM": "New_York_Mets_logo.svg",
    "NYY": "New_York_Yankees_logo.svg",
    "OAK": "Oakland_Athletics_logo.svg",
    "PHI": "Philadelphia_Phillies_logo.svg",
    "PIT": "Pittsburgh_Pirates_logo.svg",
    "SDP": "San_Diego_Padres_logo.svg",
    "SFG": "San_Francisco_Giants_logo.svg",
    "SEA": "Seattle_Mariners_logo.svg",
    "STL": "St._Louis_Cardinals_logo.svg",
    "TBR": "Tampa_Bay_Rays_logo.svg",
    "TEX": "Texas_Rangers_logo.svg",
    "TOR": "Toronto_Blue_Jays_logo.svg",
    "WSN": "Washington_Nationals_logo.svg",
}

# Using a CDN service for team logos (more reliable)
# These are publicly available team logos
TEAM_LOGO_URLS = {
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHC": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CWS": "https://a.espncdn.com/i/teamlogos/mlb/500/cws.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KCR": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAD": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYM": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYY": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SDP": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SFG": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TBR": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSN": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png",  # ESPN uses 'wsh' for Washington
}


def download_logo(team_abbr: str, logo_dir: Path, use_wikipedia: bool = False) -> bool:
    """Download a team logo.
    
    Args:
        team_abbr: Team abbreviation (e.g., "LAD", "SFG")
        logo_dir: Directory to save the logo
        use_wikipedia: If True, try Wikipedia/Wikimedia first
        
    Returns:
        True if successful, False otherwise
    """
    if team_abbr not in TEAM_LOGO_URLS:
        logger.warning(f"Unknown team abbreviation: {team_abbr}")
        return False
    
    logo_path = logo_dir / f"{team_abbr}.png"
    
    # Skip if already exists
    if logo_path.exists():
        logger.info(f"Logo already exists for {team_abbr}, skipping...")
        return True
    
    url = TEAM_LOGO_URLS[team_abbr]
    
    try:
        logger.info(f"Downloading logo for {team_abbr} from {url}")
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Save the image
        with open(logo_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Successfully downloaded logo for {team_abbr}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download logo for {team_abbr}: {e}")
        # Try alternative method using Wikipedia API
        if use_wikipedia:
            return download_from_wikipedia(team_abbr, logo_dir)
        return False
    except Exception as e:
        logger.error(f"Error saving logo for {team_abbr}: {e}")
        return False


def download_from_wikipedia(team_abbr: str, logo_dir: Path) -> bool:
    """Try to download logo from Wikipedia/Wikimedia Commons.
    
    Args:
        team_abbr: Team abbreviation
        logo_dir: Directory to save the logo
        
    Returns:
        True if successful, False otherwise
    """
    if team_abbr not in TEAM_LOGOS:
        return False
    
    logo_name = TEAM_LOGOS[team_abbr]
    # Wikimedia Commons API
    api_url = "https://commons.wikimedia.org/w/api.php"
    
    try:
        # Get image info
        params = {
            'action': 'query',
            'titles': f'File:{logo_name}',
            'prop': 'imageinfo',
            'iiprop': 'url',
            'format': 'json'
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return False
        
        page = list(pages.values())[0]
        imageinfo = page.get('imageinfo', [])
        if not imageinfo:
            return False
        
        image_url = imageinfo[0].get('url')
        if not image_url:
            return False
        
        # Download the image
        img_response = requests.get(image_url, timeout=10)
        img_response.raise_for_status()
        
        logo_path = logo_dir / f"{team_abbr}.png"
        with open(logo_path, 'wb') as f:
            f.write(img_response.content)
        
        logger.info(f"Successfully downloaded {team_abbr} logo from Wikipedia")
        return True
        
    except Exception as e:
        logger.warning(f"Wikipedia download failed for {team_abbr}: {e}")
        return False


def download_all_logos(logo_dir: Optional[Path] = None, teams: Optional[list] = None) -> Dict[str, bool]:
    """Download logos for all teams or specified teams.
    
    Args:
        logo_dir: Directory to save logos (defaults to assets/logos/)
        teams: List of team abbreviations to download (None = all teams)
        
    Returns:
        Dictionary mapping team abbreviations to success status
    """
    if logo_dir is None:
        logo_dir = project_root / "assets" / "logos"
    
    logo_dir.mkdir(parents=True, exist_ok=True)
    
    if teams is None:
        teams = list(TEAM_LOGO_URLS.keys())
    
    results = {}
    for team in teams:
        results[team] = download_logo(team, logo_dir)
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MLB team logos")
    parser.add_argument(
        "--teams",
        nargs="+",
        help="Team abbreviations to download (e.g., LAD SFG NYY). If not specified, downloads all teams."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for logos (defaults to assets/logos/)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    teams = args.teams if args.teams else None
    
    logger.info("Starting logo download...")
    results = download_all_logos(output_dir, teams)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"Logo Download Summary")
    print(f"{'='*50}")
    print(f"Successfully downloaded: {successful}/{total}")
    print(f"\nResults:")
    for team, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {team}")
    print(f"{'='*50}\n")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    exit(main())

