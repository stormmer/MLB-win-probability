"""WSGI entry point for production deployment."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Change to dashboard directory for imports
import os
os.chdir(str(Path(__file__).parent))

# Import the Dash app
from dash_app import app

# For WSGI servers (Gunicorn, uWSGI, etc.)
# Gunicorn expects a 'server' variable
server = app.server

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

