"""WSGI entry point for production deployment."""

import sys
import os
from pathlib import Path

# Get project root (parent of dashboard directory)
project_root = Path(__file__).parent.parent

# Add paths
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# Set working directory to project root
os.chdir(str(project_root))

# Import the Dash app
try:
    from dashboard.dash_app import app
except ImportError:
    # Fallback if running from dashboard directory
    from dash_app import app

# For WSGI servers (Gunicorn, uWSGI, etc.)
# Gunicorn expects a 'server' variable
server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)
