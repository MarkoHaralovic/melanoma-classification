#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
    DEVICE="cuda:0"
    echo "Running on GPU"
else
    DEVICE="cpu"
    echo "Running on CPU"
fi

python melanoma_train.py \
    --kaggle \
    --skin_color_csv "/kaggle/input/isic-2020-full-csv/ISIC_2020_full.csv" \
    --model convnext_tiny \
    --batch_size 96 \
    --epochs 10 \
    --device $DEVICE \
    --input_size 224 \
    --num_classes 2 \
    --num_workers 0 \
    --pretrained True \
    --output_dir "./melanoma_classifier_output" \
    --log_dir "./melanoma_logs" \
    --warmup_epochs 0 \
    --use_amp False \
    --mixup 0.0 \
    --update_freq 1 \
    --ohem \
    --ifw