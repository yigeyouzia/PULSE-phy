#!/usr/bin/env bash
set -euo pipefail

# Run from the PULSE repository root. Horizons run sequentially on GPU 0.
# Edit the list below to run only selected prediction lengths.
mkdir -p ./logs/Gjemnessund

for pred_len in 96 192 336 720; do
  printf '\n[Gjemnessund] Starting 720 -> %s\n' "${pred_len}"
  python -u run.py \
    --is_training 1 \
    --model_id "Gjemnessund_720_${pred_len}" \
    --model PULSE \
    --data Gjemnessund \
    --root_path ./all_datasets/ \
    --data_path gjemnessund.h5 \
    --features M \
    --seq_len 720 \
    --label_len 0 \
    --pred_len "${pred_len}" \
    --enc_in 10 \
    --time_dim 6 \
    --dsa 1 \
    --dsb 1 \
    --ksize 5 \
    --d_model 16 \
    --inv_len 96 \
    --patch_size 24 \
    --gje_window_stride 16 \
    --gje_cycle_len 96 \
    --gje_log_interval 100 \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --auxi_mode fft \
    --auxi_loss MAE \
    --module_first 1 \
    --train_epochs 30 \
    --patience 5 \
    --batch_size 16 \
    --learning_rate 0.005 \
    --lradj type3 \
    --random_seed 2024 \
    --num_workers 0 \
    --gpu 0 \
    --itr 1 2>&1 | tee "./logs/Gjemnessund/Gjemnessund_720_${pred_len}.log"
  printf "[Gjemnessund] Finished 720 -> %s\n" "${pred_len}"
done
