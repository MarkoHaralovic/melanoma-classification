#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
    DEVICE="cuda:0"
    echo "Running on GPU"
else
    DEVICE="cpu"
    echo "Running on CPU"
fi

python melanoma_train.py \
      --data_path "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge" \
      --skin_color_csv "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge/ISIC_2020_full.csv" \
      --model convnext_tiny \
      --batch_size 8 \
      --epochs 10 \
      --device $DEVICE \
      --input_size 224 \
      --num_groups 4 \
      --num_classes 2 \
      --num_workers 0 \
      --pretrained True \
      --log_dir "./melanoma_logs" \
      --warmup_epochs 1 \
      --use_amp False \
      --mixup 0.0 \
      --update_freq 1 \
      --domain_independent_loss \
      --ifw  \
      --cielab \
      --weight_decay 0.0001 \
      --lr 0.0001 \
      --output_dir "C:\lumen_melanoma_classification\melanoma-classification\melanoma_classifier_output"