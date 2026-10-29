#!/usr/bin/env bash
# Configuration source: scripts/ETTh1.sh
# Source SHA256: c26bcb6d0e98bbb52e0dc200ec95798d575f5c5722cd046a8c7b6839bae7feff
set -e

model_name=PULSE_SCF_SIMPLE
model_id_name=ETTh1
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=ETTh1.csv
data_name=ETTh1
seq_len=96

enc_in=7
dsa=1
dsb=1
time_dim=1
ksize=5
d_model=16
patch_size=24
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
freq='h'

pred_len=96
inv_len=96
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=192
inv_len=24
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=336
inv_len=24
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=720
inv_len=24
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model $model_name \
    --checkpoints "$checkpoint_dir" \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
