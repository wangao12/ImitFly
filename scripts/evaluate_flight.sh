#!/usr/bin/env bash
set -euo pipefail
python -m imitfly.cli.evaluate --checkpoint "${1:?Usage: evaluate_flight.sh CHECKPOINT WPG OUTPUT_DIR}" --wpg "${2:?Provide wing_pattern_fmech.npy}" --output "${3:?Provide a new output directory}" --seeds 0 1 2 --video
