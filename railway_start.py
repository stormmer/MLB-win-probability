#!/usr/bin/env python
"""Simple Railway startup script that definitely works."""

import os
import sys
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import and run
from dashboard.dash_app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting MLB Dashboard on port {port}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    
    app.run_server(
        debug=False,
        host="0.0.0.0",
        port=port,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )

