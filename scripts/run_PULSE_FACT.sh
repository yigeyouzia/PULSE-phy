#!/usr/bin/env bash
# Run from the PULSE-phy repository root.
set -e

echo "Starting PULSE_FACT: all 12 datasets"
bash scripts/PULSE_FACT/ETTh1.sh
bash scripts/PULSE_FACT/ETTh2.sh
bash scripts/PULSE_FACT/ETTm1.sh
bash scripts/PULSE_FACT/ETTm2.sh
bash scripts/PULSE_FACT/electricity.sh
bash scripts/PULSE_FACT/traffic.sh
bash scripts/PULSE_FACT/Weather.sh
bash scripts/PULSE_FACT/Solar.sh
bash scripts/PULSE_FACT/PEMS03.sh
bash scripts/PULSE_FACT/PEMS04.sh
bash scripts/PULSE_FACT/PEMS07.sh
bash scripts/PULSE_FACT/PEMS08.sh
