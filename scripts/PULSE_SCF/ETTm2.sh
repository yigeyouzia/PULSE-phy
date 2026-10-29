#!/usr/bin/env bash
# Configuration source: scripts/ETTm2.sh
# Source SHA256: a6ad6176020abb97ccc563e90eefe4fbfeda60c691796e76b32e659ec9bc8cf3
set -e

model_name=PULSE_SCF
model_id_name=ETTm2
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=./all_datasets/
data_path_name=ETTm2.csv
data_name=ETTm2
seq_len=96

enc_in=7
time_dim=2
batch_size=256
learning_rate=0.005
train_epochs=30
patience=5
gpu=0
freq='t'


pred_len=96
dsa=2
dsb=1
ksize=5
d_model=32
inv_len=12
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=192
dsa=2
dsb=1
ksize=3
d_model=16
inv_len=12
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=336
dsa=1
dsb=2
ksize=3
d_model=16
inv_len=48
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1


pred_len=720
dsa=1
dsb=1
ksize=3
d_model=16
inv_len=12
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
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --gpu $gpu \
    --freq $freq \
    --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
