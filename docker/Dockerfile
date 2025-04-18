FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv dos2unix \
    && ln -sf python3.10 /usr/bin/python3 \
    && pip3 install --upgrade pip

WORKDIR /melanoma-classification

RUN pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install timm==0.3.2 tensorboardX six \
    numpy==1.22 submitit fairlearn kneed torchsampler tqdm opencv-python

COPY ../ /melanoma-classification

RUN find . -type f -name "*.sh" -exec dos2unix {} \;
CMD ["/bin/bash"]