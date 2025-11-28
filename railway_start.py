#!/usr/bin/env python
"""Simple Railway startup script with error handling."""

import os
import sys
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent
os.chdir(str(project_root))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print("=" * 50)
print("Starting MLB Win Probability Dashboard")
print("=" * 50)
print(f"Working directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")

# Check if required directories exist
required_dirs = ['src', 'dashboard', 'data', 'models']
for dir_name in required_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"✓ {dir_name}/ exists")
    else:
        print(f"⚠ {dir_name}/ not found (may cause issues)")

try:
    print("\nImporting dashboard app...")
    from dashboard.dash_app import app
    print("✓ Dashboard app imported successfully")
    
    # Verify app is valid
    if not hasattr(app, 'server'):
        raise RuntimeError("app.server not found")
    print("✓ App server configured")
    
except Exception as e:
    print(f"\n✗ ERROR importing app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\nStarting server on 0.0.0.0:{port}")
    print("=" * 50)
    
    try:
        app.run_server(
            debug=False,
            host="0.0.0.0",
            port=port,
            dev_tools_ui=False,
            dev_tools_props_check=False
        )
    except Exception as e:
        print(f"\n✗ ERROR starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
