#!/bin/bash


MAMBA_ENV="skanji"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/bash/set_vars.sh"

python ./src/generate.py \
  --model_path="$CKPT_DIR/skanji" \
  --output_dir="$OUTPUT_DIR/generation" \
  --prompt="climate change" \
  --num_images=20 \
   --checkpoint=100


echo "END TIME: $(date)"
echo "DONE"