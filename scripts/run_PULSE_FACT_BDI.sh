#!/usr/bin/env bash
# Run from the PULSE-phy repository root. Traffic datasets first.
set -e

echo "Starting PULSE_FACT_BDI: all 12 datasets, 48 runs"
bash scripts/PULSE_FACT_BDI/PEMS04.sh
bash scripts/PULSE_FACT_BDI/PEMS03.sh
bash scripts/PULSE_FACT_BDI/PEMS07.sh
bash scripts/PULSE_FACT_BDI/PEMS08.sh
bash scripts/PULSE_FACT_BDI/ETTh1.sh
bash scripts/PULSE_FACT_BDI/ETTh2.sh
bash scripts/PULSE_FACT_BDI/ETTm1.sh
bash scripts/PULSE_FACT_BDI/ETTm2.sh
bash scripts/PULSE_FACT_BDI/electricity.sh
bash scripts/PULSE_FACT_BDI/traffic.sh
bash scripts/PULSE_FACT_BDI/Weather.sh
bash scripts/PULSE_FACT_BDI/Solar.sh
