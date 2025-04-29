#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
    DEVICE="cuda:0"
    echo "Running on GPU"
else
    DEVICE="cpu"
    echo "Running on CPU"
fi

python melanoma_train.py \
   --data_path "./isic2020_challenge" \
   --skin_color_csv "./isic2020_challenge/ISIC_2020_full.csv" \
   --model convnext_tiny \
   --batch_size 8 \
   --epochs 10 \
   --device $DEVICE \
   --input_size 224 \
   --num_classes 2 \
   --num_groups 4 \
   --num_workers 0 \
   --pretrained True \
   --log_dir "./melanoma_logs" \
   --ifw  \
   --checkpoint '.\weights\best_model_domain_discriminative.pth' \
   --output_dir ".\melanoma_classifier_output" \
   --test 