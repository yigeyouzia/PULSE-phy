#!/usr/bin/env bash
# Run from the PULSE-phy repository root.
set -e

echo "Starting PULSE_SCF: all 12 datasets"
bash scripts/PULSE_SCF/ETTh1.sh
bash scripts/PULSE_SCF/ETTh2.sh
bash scripts/PULSE_SCF/ETTm1.sh
bash scripts/PULSE_SCF/ETTm2.sh
bash scripts/PULSE_SCF/electricity.sh
bash scripts/PULSE_SCF/traffic.sh
bash scripts/PULSE_SCF/Weather.sh
bash scripts/PULSE_SCF/Solar.sh
bash scripts/PULSE_SCF/PEMS03.sh
bash scripts/PULSE_SCF/PEMS04.sh
bash scripts/PULSE_SCF/PEMS07.sh
bash scripts/PULSE_SCF/PEMS08.sh

echo "Starting PULSE_SOFTS: all 12 datasets"
bash scripts/PULSE_SOFTS/ETTh1.sh
bash scripts/PULSE_SOFTS/ETTh2.sh
bash scripts/PULSE_SOFTS/ETTm1.sh
bash scripts/PULSE_SOFTS/ETTm2.sh
bash scripts/PULSE_SOFTS/electricity.sh
bash scripts/PULSE_SOFTS/traffic.sh
bash scripts/PULSE_SOFTS/Weather.sh
bash scripts/PULSE_SOFTS/Solar.sh
bash scripts/PULSE_SOFTS/PEMS03.sh
bash scripts/PULSE_SOFTS/PEMS04.sh
bash scripts/PULSE_SOFTS/PEMS07.sh
bash scripts/PULSE_SOFTS/PEMS08.sh

echo "Starting PULSE_SCF_SIMPLE: all 12 datasets"
bash scripts/PULSE_SCF_SIMPLE/ETTh1.sh
bash scripts/PULSE_SCF_SIMPLE/ETTh2.sh
bash scripts/PULSE_SCF_SIMPLE/ETTm1.sh
bash scripts/PULSE_SCF_SIMPLE/ETTm2.sh
bash scripts/PULSE_SCF_SIMPLE/electricity.sh
bash scripts/PULSE_SCF_SIMPLE/traffic.sh
bash scripts/PULSE_SCF_SIMPLE/Weather.sh
bash scripts/PULSE_SCF_SIMPLE/Solar.sh
bash scripts/PULSE_SCF_SIMPLE/PEMS03.sh
bash scripts/PULSE_SCF_SIMPLE/PEMS04.sh
bash scripts/PULSE_SCF_SIMPLE/PEMS07.sh
bash scripts/PULSE_SCF_SIMPLE/PEMS08.sh
