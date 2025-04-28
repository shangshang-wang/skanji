#!/bin/bash


MAMBA_ENV="skanji"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

PY_SCRIPT="./scripts/train/train_text_to_image.py"

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" \
  --pretrained_model_name_or_path="bguisard/stable-diffusion-nano-2-1" \
  --train_data_dir="./assets/data/chinese_images/white_background_original" \
  --use_ema \
  --resolution=128 --center_crop \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=500 \
  --snr_gamma=5.0 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./train_chinese_from_scratch_original_low_res_128" \
  --validation_prompt="spring" \
  --report_to="wandb" \
  --checkpointing_steps=100 --checkpoints_total_limit=20 \
  --validation_epochs=1 \
  --use_8bit_adam \
  --from_scratch \
  --resume_from_checkpoint="latest"


echo "END TIME: $(date)"
echo "DONE"