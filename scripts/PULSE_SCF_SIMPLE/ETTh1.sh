#!/usr/bin/env bash
# Source layout/configuration: scripts/ETTh1.sh. Run from the repository root.
# Source SHA256: c26bcb6d0e98bbb52e0dc200ec95798d575f5c5722cd046a8c7b6839bae7feff
# Default prediction lengths: 96 192 336 720. Use PRED_LENS to select a subset.
# Log name: <ModelName>_<seq_len>_<pred_len>.log; run identity stays in the directory.
# Fixed model; this script does not read MODEL_NAME.
# SCF-specific dimensions use independent model defaults; no new run.py flags.
set -euo pipefail
set -o noclobber

model_id_name=ETTh1
model_name=PULSE_SCF_SIMPLE
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
read -r -a scf_pred_lens <<< "${PRED_LENS:-96 192 336 720}"

case "$model_name" in
    PULSE|PULSE_SOFTS|PULSE_SCF_SIMPLE|PULSE_SCF) ;;
    *) printf 'Unsupported model: %s\n' "$model_name" >&2; exit 2 ;;
esac
case "$run_tag" in
    ""|*[!a-zA-Z0-9._-]*) printf 'RUN_TAG must use letters, digits, dots, underscores or hyphens.\n' >&2; exit 2 ;;
esac
for horizon in "${scf_pred_lens[@]}"; do
    case "$horizon" in
        96|192|336|720) ;;
        *) printf 'Unsupported prediction length for ETTh1: %s\n' "$horizon" >&2; exit 2 ;;
    esac
done
if [ ! -f run.py ]; then
    printf 'Run from the PULSE-phy repository root.\n' >&2
    exit 2
fi

scf_selected() {
    local requested
    for requested in "${scf_pred_lens[@]}"; do
        if [ "$requested" = "$1" ]; then return 0; fi
    done
    return 1
}

scf_log_path() {
    printf '%s/%s_%s_%s.log' "$dir" "$model_name" "$seq_len" "$1"
}

scf_announce_log() {
    local absolute_log="${PWD}/${1#./}"
    printf '[%s] %s: %s -> %s\n' "$model_name" "$data_name" "$seq_len" "$pred_len"
    printf 'Log: %s\n' "$absolute_log"
    printf 'Watch: tail -f %q\n' "$absolute_log"
    printf 'Checkpoint root: %s/%s\n' "$PWD" "${checkpoint_dir#./}"
}

dir=./logs/${model_name}/${model_id_name}/${run_tag}
checkpoint_dir=./checkpoints/${model_name}/
mkdir -p "$dir"

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
if scf_selected "$pred_len"; then
    log_file=$(scf_log_path "$pred_len")
    scf_announce_log "$log_file"
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${run_tag}" \
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
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=192
inv_len=24
if scf_selected "$pred_len"; then
    log_file=$(scf_log_path "$pred_len")
    scf_announce_log "$log_file"
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${run_tag}" \
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
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=336
inv_len=24
if scf_selected "$pred_len"; then
    log_file=$(scf_log_path "$pred_len")
    scf_announce_log "$log_file"
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${run_tag}" \
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
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=720
inv_len=24
if scf_selected "$pred_len"; then
    log_file=$(scf_log_path "$pred_len")
    scf_announce_log "$log_file"
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id "${model_name}_${model_id_name}_${seq_len}_${pred_len}_${run_tag}" \
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
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi
