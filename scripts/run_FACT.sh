#!/usr/bin/env bash
# Run from the PULSE-phy repository root.
set -e

bash scripts/FACT/Gjemnessund.sh
bash scripts/FACT/ETTh1.sh
bash scripts/FACT/ETTh2.sh
bash scripts/FACT/ETTm1.sh
bash scripts/FACT/ETTm2.sh
bash scripts/FACT/electricity.sh
bash scripts/FACT/traffic.sh
bash scripts/FACT/Weather.sh
bash scripts/FACT/Solar.sh
bash scripts/FACT/PEMS03.sh
bash scripts/FACT/PEMS04.sh
bash scripts/FACT/PEMS07.sh
bash scripts/FACT/PEMS08.sh
