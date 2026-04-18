#!/bin/bash
# Monthly timelapse portrait pipeline runner.
# Called by launchd. Can also be run manually: bash run_monthly.sh

cd /Users/ian/Projects/timelapse_portrait
export PATH="/Users/ian/.local/bin:$PATH"

echo "=== Timelapse auto run: $(date) ==="
poetry run timelapse auto
echo "=== Finished: $(date) ==="
