#!/usr/bin/env bash
# Configuration source: scripts/Weather.sh
# Source SHA256: 9656df40519aa1546003a5c9a0cdbf5b327223a443c69bb7b37c9e1fcd41f2dc
set -e

model_name=PULSE_SOFTS
model_id_name=weather
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=weather.csv
data_name=Weather
seq_len=96
enc_in=21

batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
rec_lambda=0
auxi_lambda=1
itr=1
freq='h'

pred_len=96
time_dim=2
dsa=4
dsb=1
ksize=5
d_model=32
inv_len=24
patch_size=24

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
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=192
time_dim=2
dsa=4
dsb=1
ksize=5
d_model=16
inv_len=48
patch_size=6

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
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=336
time_dim=1
dsa=4
dsb=2
ksize=3
d_model=32
inv_len=72
patch_size=24

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
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=720
time_dim=1
dsa=2
dsb=1
ksize=3
d_model=32
inv_len=96
patch_size=6

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
    --rec_lambda $rec_lambda \
    --auxi_lambda $auxi_lambda \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --itr $itr --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
