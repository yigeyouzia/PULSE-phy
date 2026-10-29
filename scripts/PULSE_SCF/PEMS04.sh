#!/usr/bin/env bash
# Configuration source: scripts/PEMS04.sh
# Source SHA256: ada24204b85e36e94bdc204b984310cfc340d0c4a069e5b8ef8d9da40b03eb7b
set -e

model_name=PULSE_SCF
model_id_name=PEMS04
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=PEMS04.npz
data_name=PEMS
seq_len=96

enc_in=307
time_dim=1
train_epochs=30
patience=5
gpu=0
batch_size=32
learning_rate=0.005
rec_lambda=0
auxi_lambda=1
random_seed=2024

pred_len=12
inv_len=96
dsa=4
dsb=1
ksize=5
d_model=32
patch_size=4

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=24
inv_len=96
dsa=2
dsb=1
ksize=5
d_model=16
patch_size=8

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=48
inv_len=24
dsa=4
dsb=1
ksize=5
d_model=16
patch_size=12

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=96
inv_len=48
dsa=4
dsb=1
ksize=5
d_model=16
patch_size=12

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
    --random_seed $random_seed \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
