#!/bin/bash
# Force update all packages in the virtual environment using uv

set -e

echo "Updating all packages using uv..."

# Sync and upgrade all dependencies to their latest versions
uv sync --upgrade

echo "All packages have been updated successfully!"
