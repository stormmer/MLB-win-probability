"""Gunicorn configuration for production deployment."""

import multiprocessing
import os

# Server socket
port = int(os.environ.get("PORT", 8050))
bind = f"0.0.0.0:{port}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "mlb_win_probability_dashboard"

