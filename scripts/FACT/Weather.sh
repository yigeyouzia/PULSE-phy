#!/usr/bin/env bash
# Common data/budget source: scripts/PULSE/Weather.sh
# Source SHA256: 9a97be26840bcd5ade244d00e19fa13ba4c9245cbf3763967dd5df06efb8a06a
# Complete FACT; MSE/no SAM; model configuration differs from PULSE Router settings.
set -e

model_name=FACT
model_id_name=weather
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=weather.csv
data_name=Weather
enc_in=21
time_dim=2
freq=h

fact_d_model=512
fact_d_ff=2048
fact_core=0.5
fact_num_kernels=4
fact_dropout=0.1
fact_use_norm=1

seq_len=96
pred_len=96
time_dim=2
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --freq $freq \
    --fact_d_model $fact_d_model \
    --fact_d_ff $fact_d_ff \
    --fact_core $fact_core \
    --fact_dilation 1 2 3 2 1 \
    --fact_num_kernels $fact_num_kernels \
    --fact_dropout $fact_dropout \
    --fact_use_norm $fact_use_norm \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --loss mse \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --pct_start 0.3 \
    --random_seed 2024 \
    --num_workers 10 \
    --gpu 0 \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

seq_len=96
pred_len=192
time_dim=2
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --freq $freq \
    --fact_d_model $fact_d_model \
    --fact_d_ff $fact_d_ff \
    --fact_core $fact_core \
    --fact_dilation 1 2 3 2 1 \
    --fact_num_kernels $fact_num_kernels \
    --fact_dropout $fact_dropout \
    --fact_use_norm $fact_use_norm \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --loss mse \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --pct_start 0.3 \
    --random_seed 2024 \
    --num_workers 10 \
    --gpu 0 \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

seq_len=96
pred_len=336
time_dim=1
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --freq $freq \
    --fact_d_model $fact_d_model \
    --fact_d_ff $fact_d_ff \
    --fact_core $fact_core \
    --fact_dilation 1 2 3 2 1 \
    --fact_num_kernels $fact_num_kernels \
    --fact_dropout $fact_dropout \
    --fact_use_norm $fact_use_norm \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --loss mse \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --pct_start 0.3 \
    --random_seed 2024 \
    --num_workers 10 \
    --gpu 0 \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

seq_len=96
pred_len=720
time_dim=1
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 0 \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --freq $freq \
    --fact_d_model $fact_d_model \
    --fact_d_ff $fact_d_ff \
    --fact_core $fact_core \
    --fact_dilation 1 2 3 2 1 \
    --fact_num_kernels $fact_num_kernels \
    --fact_dropout $fact_dropout \
    --fact_use_norm $fact_use_norm \
    --rec_lambda 1 \
    --auxi_lambda 0 \
    --loss mse \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --pct_start 0.3 \
    --random_seed 2024 \
    --num_workers 10 \
    --gpu 0 \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
