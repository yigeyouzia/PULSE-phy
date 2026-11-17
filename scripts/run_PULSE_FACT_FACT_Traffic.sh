#!/usr/bin/env bash
# Run from the PULSE-phy repository root.
set -e

echo "Starting PULSE_FACT: four FACT traffic datasets, 16 experiments"
bash scripts/PULSE_FACT/PEMSD7.sh
bash scripts/PULSE_FACT/PEMS-BAY.sh
bash scripts/PULSE_FACT/Traffic2.sh
bash scripts/PULSE_FACT/METR-LA.sh
