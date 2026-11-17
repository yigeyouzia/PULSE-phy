#!/usr/bin/env bash
# Budget reference: FACT.sh in the official wanghq21/FACT repository.
# PULSE_FACT model widths follow PEMS04.sh at 96-to-12.
set -e

model_name=PULSE_FACT
model_id_name=METR-LA
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=METR-LA.csv
data_name=FACTTraffic
seq_len=96
enc_in=207
freq=5min
time_dim=5

train_epochs=15
patience=3
gpu=0
batch_size=32
learning_rate=0.001
lradj=type1
rec_lambda=0
auxi_lambda=1
random_seed=2024

d_model=32
dsa=4
dsb=1
ksize=5
inv_len=288
patch_size=12

pred_len=12
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model "$model_name" \
    --checkpoints "$checkpoint_dir" \
    --data "$data_name" \
    --features M \
    --embed timeF \
    --freq "$freq" \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --time_dim "$time_dim" \
    --dsa "$dsa" \
    --dsb "$dsb" \
    --ksize "$ksize" \
    --d_model "$d_model" \
    --inv_len "$inv_len" \
    --patch_size "$patch_size" \
    --rec_lambda "$rec_lambda" \
    --auxi_lambda "$auxi_lambda" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --lradj "$lradj" \
    --gpu "$gpu" \
    --random_seed "$random_seed" \
    --itr 1 --batch_size "$batch_size" --learning_rate "$learning_rate" > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=24
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model "$model_name" \
    --checkpoints "$checkpoint_dir" \
    --data "$data_name" \
    --features M \
    --embed timeF \
    --freq "$freq" \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --time_dim "$time_dim" \
    --dsa "$dsa" \
    --dsb "$dsb" \
    --ksize "$ksize" \
    --d_model "$d_model" \
    --inv_len "$inv_len" \
    --patch_size "$patch_size" \
    --rec_lambda "$rec_lambda" \
    --auxi_lambda "$auxi_lambda" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --lradj "$lradj" \
    --gpu "$gpu" \
    --random_seed "$random_seed" \
    --itr 1 --batch_size "$batch_size" --learning_rate "$learning_rate" > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=48
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model "$model_name" \
    --checkpoints "$checkpoint_dir" \
    --data "$data_name" \
    --features M \
    --embed timeF \
    --freq "$freq" \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --time_dim "$time_dim" \
    --dsa "$dsa" \
    --dsb "$dsb" \
    --ksize "$ksize" \
    --d_model "$d_model" \
    --inv_len "$inv_len" \
    --patch_size "$patch_size" \
    --rec_lambda "$rec_lambda" \
    --auxi_lambda "$auxi_lambda" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --lradj "$lradj" \
    --gpu "$gpu" \
    --random_seed "$random_seed" \
    --itr 1 --batch_size "$batch_size" --learning_rate "$learning_rate" > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=96
echo "Log: ${PWD}/${dir#./}/${model_name}_${seq_len}_${pred_len}.log"
python -u run.py \
    --is_training 1 \
    --root_path "$root_path_name" \
    --data_path "$data_path_name" \
    --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_id}" \
    --model "$model_name" \
    --checkpoints "$checkpoint_dir" \
    --data "$data_name" \
    --features M \
    --embed timeF \
    --freq "$freq" \
    --seq_len "$seq_len" \
    --label_len 0 \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --time_dim "$time_dim" \
    --dsa "$dsa" \
    --dsb "$dsb" \
    --ksize "$ksize" \
    --d_model "$d_model" \
    --inv_len "$inv_len" \
    --patch_size "$patch_size" \
    --rec_lambda "$rec_lambda" \
    --auxi_lambda "$auxi_lambda" \
    --train_epochs "$train_epochs" \
    --patience "$patience" \
    --lradj "$lradj" \
    --gpu "$gpu" \
    --random_seed "$random_seed" \
    --itr 1 --batch_size "$batch_size" --learning_rate "$learning_rate" > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
