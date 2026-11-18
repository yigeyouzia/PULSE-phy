#!/usr/bin/env bash
# Run from the PULSE-phy repository root. Each dataset runs H=12, then H=96.
set -e

bash scripts/PULSE_FACT_A00/PEMS04.sh
bash scripts/PULSE_FACT_A00/PEMS08.sh
bash scripts/PULSE_FACT_A10/PEMS04.sh
bash scripts/PULSE_FACT_A10/PEMS08.sh
bash scripts/PULSE_FACT_A01/PEMS04.sh
bash scripts/PULSE_FACT_A01/PEMS08.sh
