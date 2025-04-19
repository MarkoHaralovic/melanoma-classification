#!/bin/bash

if [ -x "$(command -v nvidia-smi)" ]; then
   DEVICE="cuda:0"
   echo "Running on GPU"
else
   DEVICE="cpu"
   echo "Running on CPU"
fi

python -m torch.distributed.launch --nproc_per_node=4 melanoma_train \
   --data_path "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge" \
   --skin_color_csv "C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge/ISIC_2020_full.csv" \
   --model dinov2_vit_small \
   --in_22k False \
   --batch_size 8 \
   --epochs 10 \
   --device $DEVICE \
   --freeze_model True \
   --input_size 224 \
   --num_classes 2 \
   --num_workers 4 \
   --pretrained True \
   --log_dir "./melanoma_logs" \
   --warmup_epochs 0 \
   --use_amp False \
   --lr 0.01 \
   --mixup 0.0 \
   --update_freq 1 \
   --ifw  \
   --recall_ce \
   --weight_decay 0.0001 \
   --output_dir "C:\lumen_melanoma_classification\melanoma-classification\melanoma_classifier_output" 