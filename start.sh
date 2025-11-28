#!/bin/bash
# Startup script for Railway deployment

# Set working directory
cd /app

# Install dependencies if needed (Railway should do this, but just in case)
pip install -r requirements.txt

# Start the server
gunicorn --config gunicorn_config.py dashboard.wsgi:server

