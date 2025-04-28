# Skanji

Practicing project for training Stable Diffusion models on Japanese Kanji and Chinese Hanzi.

## Quick Start

Env setup

```bash
conda update -n base -c defaults conda -y
conda install -n base -c conda-forge mamba -y
mamba shell init --shell bash --root-prefix=~/.local/share/mamba

mamba create -n skanji python=3.10 -y && mamba activate skanji
./scripts/set/set_env.sh && mamba deactivate
```

Model training

```bash
# for Japanese Kanji
./scripts/train/full_finetune.sh # full-parameter finetune
./scripts/train/lora_finetune.sh # lora-based finetune
./scripts/train/train_from_scratch.sh # train from scratch
./scripts/train/train_from_scratch_low_res.sh # train from scratch with low res images

# for Chinese Hanzi extension
./scripts/train/train_from_scratch_chinese.sh # train from scratch
./scripts/train/train_from_scratch_chinese_low_res.sh # train from scratch with low res images
```

Model evaluation

```bash
./scripts/inference/run_inference.sh
```

