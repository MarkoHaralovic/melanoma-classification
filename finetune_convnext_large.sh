#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
    DEVICE="cuda:0"
    echo "Running on GPU"
else
    DEVICE="cpu"
    echo "Running on CPU"
fi

CONFIG_FILE=""
EXTRA_ARGS=""

python melanoma_train.py \
    --data_path "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge" \
    --skin_color_csv "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge/ISIC_2020_full.csv" \
    --model convnext_large \
    --drop_path 0.95 \
    --input_size 224 \
    --lr 5e-5 \
    --update_freq 2 \
    --warmup_epochs 0 \
    --weight_decay 1e-8 \
    --layer_decay 0.7 \
    --cutmix 0.0 \
    --mixup 0.0 \
    --batch_size 32 \
    --epochs 10 \
    --device $DEVICE \
    --num_groups 4 \
    --num_classes 2 \
    --num_workers 0 \
    --pretrained True \
    --log_dir "./melanoma_logs" \
    --use_amp False \
    --domain_independent_loss \
    --ifw  \
    --cielab \
    --output_dir "C:\lumen_melanoma_classification\melanoma-classification\melanoma_classifier_output"