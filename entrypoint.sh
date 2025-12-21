#!/bin/bash

# Chainlytics Backend ML System - Entrypoint Script

# Exit on any error
set -e

# Print startup message
echo "Starting Chainlytics Backend ML System..."
echo "Timestamp: $(date)"

# Check if required directories exist, create if not
echo "Checking directories..."
mkdir -p data/logs data/models registry/models registry/policies config

# Check if config files exist, copy defaults if not
if [ ! -f "config/env.yaml" ]; then
    echo "Creating default config files..."
    # Config files would be created by the application if needed
fi

# Set up logging directory
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

# Print system info
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Available services:"
ls -la services/

# If command is provided, execute it
if [ $# -gt 0 ]; then
    echo "Executing command: $*"
    exec "$@"
else
    # Default behavior - start the decision orchestrator
    echo "Starting decision orchestrator..."
    exec python inference/decision_orchestrator.py
fi