#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create directories
mkdir -p "$PROJECT_DIR/models"

# Create venv and install dependencies
python3 -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"
pip install -r "$PROJECT_DIR/requirements.txt"
