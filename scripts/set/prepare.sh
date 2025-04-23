#!/bin/bash


echo "Prepare the data for training ..."
python ./assets/data/preprocess_kanjivg2png.py

echo "Convert to jsonl ..."
python ./assets/data/preprocess_png2jsonl.py