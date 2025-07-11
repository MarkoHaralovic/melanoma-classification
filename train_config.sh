#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
   DEVICE="cuda:0"
   echo "Running on GPU"
else
   DEVICE="cpu"
   echo "Running on CPU"
fi

python melanoma_train.py \
   --config "configs/dino_vit_small.yaml" \
   --data_path "./isic2020_challenge" \
   --skin_color_csv "./isic2020_challenge/ISIC_2020_full.csv" \
   --device $DEVICE \
   --num_workers 4 \
   --log_dir "./melanoma_logs" \
   --output_dir "./melanoma_classifier_output"