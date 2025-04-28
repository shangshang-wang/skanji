#!/bin/bash


MAMBA_ENV="skanji"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

PY_SCRIPT="./scripts/train/train_text_to_image_lora.py"
RESOLUTION=512
# ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --train_data_dir "./assets/data/kanji_images" \
    --output_dir "${OUTPUT_DIR}/resolution_${RESOLUTION}" \
    --lora_rank 16 \
    --resolution "${RESOLUTION}" --center_crop \
    --train_batch_size 16 \
    --gradient_accumulation_steps 4 --gradient_checkpointing \
    --num_train_epochs 200 \
    --snr_gamma 5.0 \
    --max_grad_norm 1 \
    --learning_rate 1e-04 --use_8bit_adam --lr_scheduler "constant" --lr_warmup_steps 50 \
    --validation_prompt "abandon" \
    --checkpointing_steps 200 --checkpoints_total_limit 20 \
    --resume_from_checkpoint="latest" \
    --report_to "wandb" \
    --enable_xformers_memory_efficient_attention
#     --max_train_steps=300000


echo "END TIME: $(date)"
echo "DONE"