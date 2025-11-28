#!/bin/bash
# Railway startup script

# Get the port from Railway
PORT=${PORT:-8050}

# Start with gunicorn
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    dashboard.wsgi:server

