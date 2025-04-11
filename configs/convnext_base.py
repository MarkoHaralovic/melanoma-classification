# model params
model='convnext_base'
drop_path=0.8
input_size=384
batch_size=32
lr=5e-5
update_freq=2
warmup_epochs=0
epochs=10
weight_decay=1e-8
layer_decay=0.7
head_init_scale=0.001
cutmix=0
mixup=0

# other params
skin_color_csv="C:/lumen_melanoma_classification/melanoma-classification/isic2020_challenge/ISIC_2020_full.csv"
num_groups=4
num_classes=2
num_workers=0
pretrained=True
log_dir="./melanoma_logs"
use_amp=False
domain_independent_loss=True
ifw=True
cielab=True
output_dir="C:/lumen_melanoma_classification/melanoma-classification/melanoma_classifier_output"