#!/usr/bin/env bash
# Configuration source: scripts/Gjemnessund.sh (working-tree version, 2026-11-04).
# Source SHA256: 37ac49b1ec835afa83a28347e61009938620a7c91bf545e51383db6bfc857149
# Models and the original bridge script are unchanged; see notes/Gjemnessund扩展模型接入计划与通用方法.md.
set -e

model_name=PULSE_SCF
model_id_name=Gjemnessund
dir=./logs/${model_name}/${model_id_name}
checkpoint_dir=./checkpoints/${model_name}/
experiment_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$dir"
echo "Checkpoint root: ${PWD}/${checkpoint_dir#./}"

root_path_name=${GJE_ROOT_PATH:-./all_datasets/}
data_path_name=gjemnessund.h5
data_name=Gjemnessund
seq_len=720
label_len=0
enc_in=10
time_dim=6
dsa=1
dsb=1
ksize=5
d_model=16
inv_len=96
patch_size=24
gje_window_stride=16
gje_cycle_len=96
gje_log_interval=100
batch_size=16
learning_rate=0.005
train_epochs=30
patience=5
random_seed=2024
num_workers=0
gpu=0

pred_len=96
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
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --gje_window_stride $gje_window_stride \
    --gje_cycle_len $gje_cycle_len \
    --gje_log_interval $gje_log_interval \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --auxi_mode fft \
    --auxi_type complex \
    --auxi_loss MAE \
    --module_first 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --random_seed $random_seed \
    --num_workers $num_workers \
    --gpu $gpu \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=192
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
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --gje_window_stride $gje_window_stride \
    --gje_cycle_len $gje_cycle_len \
    --gje_log_interval $gje_log_interval \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --auxi_mode fft \
    --auxi_type complex \
    --auxi_loss MAE \
    --module_first 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --random_seed $random_seed \
    --num_workers $num_workers \
    --gpu $gpu \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=336
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
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --gje_window_stride $gje_window_stride \
    --gje_cycle_len $gje_cycle_len \
    --gje_log_interval $gje_log_interval \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --auxi_mode fft \
    --auxi_type complex \
    --auxi_loss MAE \
    --module_first 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --random_seed $random_seed \
    --num_workers $num_workers \
    --gpu $gpu \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1

pred_len=720
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
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --time_dim $time_dim \
    --dsa $dsa \
    --dsb $dsb \
    --ksize $ksize \
    --d_model $d_model \
    --inv_len $inv_len \
    --patch_size $patch_size \
    --gje_window_stride $gje_window_stride \
    --gje_cycle_len $gje_cycle_len \
    --gje_log_interval $gje_log_interval \
    --rec_lambda 0 \
    --auxi_lambda 1 \
    --auxi_mode fft \
    --auxi_type complex \
    --auxi_loss MAE \
    --module_first 1 \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --lradj type3 \
    --random_seed $random_seed \
    --num_workers $num_workers \
    --gpu $gpu \
    --itr 1 > "$dir/${model_name}_${seq_len}_${pred_len}.log" 2>&1
