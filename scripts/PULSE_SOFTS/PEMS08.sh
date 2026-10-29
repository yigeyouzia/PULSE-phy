#!/usr/bin/env bash
# Source layout/configuration: scripts/PEMS08.sh. Run from the repository root.
# Source SHA256: 0a33d14930d50f4147d9ec1206c3a8d3555656701068f2fc175a7fe4e6477d4e
# Default prediction lengths: 12 24 48 96. Use PRED_LENS to select a subset.
# Log name: <ModelName>_<seq_len>_<pred_len>.log; run identity stays in the directory.
# Fixed model; this script does not read MODEL_NAME.
# SCF-specific dimensions use independent model defaults; no new run.py flags.
set -euo pipefail
set -o noclobber

model_id_name=PEMS08
model_name=PULSE_SOFTS
run_tag=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
read -r -a scf_pred_lens <<< "${PRED_LENS:-12 24 48 96}"

case "$model_name" in
    PULSE|PULSE_SOFTS|PULSE_SCF_SIMPLE|PULSE_SCF) ;;
    *) printf 'Unsupported model: %s\n' "$model_name" >&2; exit 2 ;;
esac
case "$run_tag" in
    ""|*[!a-zA-Z0-9._-]*) printf 'RUN_TAG must use letters, digits, dots, underscores or hyphens.\n' >&2; exit 2 ;;
esac
for horizon in "${scf_pred_lens[@]}"; do
    case "$horizon" in
        12|24|48|96) ;;
        *) printf 'Unsupported prediction length for PEMS08: %s\n' "$horizon" >&2; exit 2 ;;
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
data_path_name=PEMS08.npz
data_name=PEMS
seq_len=96


enc_in=170
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
inv_len=32
dsa=8
dsb=2
ksize=3
d_model=32
patch_size=12

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
        --rec_lambda $rec_lambda \
        --auxi_lambda $auxi_lambda \
        --train_epochs $train_epochs \
        --patience $patience \
        --gpu $gpu \
        --random_seed $random_seed \
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=24
inv_len=12
dsa=1
dsb=2
ksize=3
d_model=32
patch_size=24

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
        --rec_lambda $rec_lambda \
        --auxi_lambda $auxi_lambda \
        --train_epochs $train_epochs \
        --patience $patience \
        --gpu $gpu \
        --random_seed $random_seed \
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=48
inv_len=24
dsa=4
dsb=2
ksize=5
d_model=16
patch_size=4

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
        --rec_lambda $rec_lambda \
        --auxi_lambda $auxi_lambda \
        --train_epochs $train_epochs \
        --patience $patience \
        --gpu $gpu \
        --random_seed $random_seed \
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi


pred_len=96
inv_len=12
dsa=1
dsb=1
ksize=5
d_model=16
patch_size=4

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
        --rec_lambda $rec_lambda \
        --auxi_lambda $auxi_lambda \
        --train_epochs $train_epochs \
        --patience $patience \
        --gpu $gpu \
        --random_seed $random_seed \
        --itr 1 --batch_size $batch_size --learning_rate $learning_rate > "$log_file" 2>&1
fi
