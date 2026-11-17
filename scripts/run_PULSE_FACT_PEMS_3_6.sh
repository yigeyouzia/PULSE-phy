#!/usr/bin/env bash
# Run from the PULSE-phy repository root.
set -e

echo "Starting PULSE_FACT: PEMS 3- and 6-step forecasts"
bash scripts/PULSE_FACT/PEMS_3_6/PEMS03.sh
bash scripts/PULSE_FACT/PEMS_3_6/PEMS04.sh
bash scripts/PULSE_FACT/PEMS_3_6/PEMS07.sh
bash scripts/PULSE_FACT/PEMS_3_6/PEMS08.sh
