#!/usr/bin/env python
"""Simple startup script for Railway deployment."""

import os
import sys
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).parent
os.chdir(str(project_root))

# Add src to path
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Change to dashboard directory
os.chdir(str(project_root / "dashboard"))

# Import and run
from dash_app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)

