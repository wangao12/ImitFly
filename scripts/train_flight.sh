#!/usr/bin/env bash
set -euo pipefail
python -m imitfly.cli.train --config configs/imitfly_12_demos.json --data "${1:?Usage: train_flight.sh DATA_DIR OUTPUT_DIR}" --output "${2:?Provide a new output directory}"
