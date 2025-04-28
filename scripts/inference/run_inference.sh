#!/bin/bash


MAMBA_ENV="skanji"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"


TRAIN_TYPE="train_from_scratch_512"
CKPT_LIST=("13000")
#TRAIN_TYPE="train_from_scratch_chinese_512"
#CKPT_LIST=("4000")
BASE_MODEL="CompVis/stable-diffusion-v1-4"

#TRAIN_TYPE="train_from_scratch_low_res_128"
#CKPT_LIST=("20000")
#TRAIN_TYPE="train_from_scratch_chinese_low_res_128"
#CKPT_LIST=("25000")
#BASE_MODEL="bguisard/stable-diffusion-nano-2-1"

for CKPT in "${CKPT_LIST[@]}"; do
    echo "➡️  Sampling with checkpoint $CKPT"
    python ./scripts/inference/inference.py \
        --base_model "$BASE_MODEL" \
        --out_dir "./assets/generation/${TRAIN_TYPE}" \
        --ckpt_root "${CKPT_DIR}/${TRAIN_TYPE}" \
        --ckpt "$CKPT" \
        --prompts "fish" "water" "sakana" "sea"
done

echo "END TIME: $(date)"
echo "DONE"
