"""WSGI entry point for production deployment."""

import sys
import os
from pathlib import Path

# Get project root (parent of dashboard directory)
project_root = Path(__file__).parent.parent

# Add paths - order matters!
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Set working directory to project root
os.chdir(str(project_root))

# Import the Dash app with error handling
try:
    # Try importing from dashboard.dash_app (when running from project root)
    from dashboard.dash_app import app
except ImportError as e1:
    try:
        # Fallback: try importing directly (when running from dashboard directory)
        from dash_app import app
    except ImportError as e2:
        print(f"Import error (method 1): {e1}")
        print(f"Import error (method 2): {e2}")
        print(f"Python path: {sys.path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        raise

# For WSGI servers (Gunicorn, uWSGI, etc.)
# Gunicorn expects a 'server' variable
if not hasattr(app, 'server'):
    raise RuntimeError("Dash app.server not found - app may not be initialized correctly")

server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting server on port {port}")
    app.run_server(debug=False, host="0.0.0.0", port=port)
