#!/bin/bash


MAMBA_ENV="skanji"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

PY_SCRIPT="./scripts/train/train_text_to_image.py"

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="./assets/data/kanji_images" \
  --use_ema \
  --resolution=512 --center_crop \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=400 \
  --snr_gamma=5.0 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="./finetune_full" \
  --validation_prompt="abandon" \
  --report_to="wandb" \
  --checkpointing_steps=100 --checkpoints_total_limit=20 \
  --validation_epochs=1 \
  --use_8bit_adam \
  --resume_from_checkpoint="latest"


echo "END TIME: $(date)"
echo "DONE"