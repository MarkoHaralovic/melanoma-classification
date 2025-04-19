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
   --model convnextv2_atto \
   --batch_size 8 \
   --epochs 10 \
   --device $DEVICE \
   --input_size 224 \
   --num_classes 2 \
   --num_workers 0 \
   --pretrained True \
   --log_dir "./melanoma_logs" \
   --ifw  \
   --checkpoint 'C:\lumen_melanoma_classification\melanoma-classification\melanoma_classifier_output\20250418_030204_convnextv2_atto_bs8_ifw\checkpoint_epoch_3.pth' \
   --output_dir "C:\lumen_melanoma_classification\melanoma-classification\melanoma_classifier_output" \
   --test 