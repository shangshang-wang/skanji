#!/bin/bash
# python 3.10 + cuda 11.8.0
# the execution order the following commands matter

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda clean -a -y # conda for traditional and reliable setup
mamba clean -a -y # mamba for smart and efficient setup
pip install --upgrade pip
pip cache purge

# cuda, gcc/g++, torch
mamba install cuda -c nvidia/label/cuda-11.8.0 -y
mamba install gcc gxx -c conda-forge -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# xformers
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade diffusers[torch]
pip install transformers==4.51.1
pip install accelerate==1.6.0
pip install wandb