# Installation

We provide installation instructions.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n melanoma python=3.10 -y
conda activate melanoma
```

Run:
```
pip install -r requirements.txt
```

For full control over torch and torchvision versions, you can install them manually.

For CPU only, run:
```
pip install \
   torch==2.2.0+cpu \
   torchvision==0.17.0+cpu \
   torchaudio==2.2.0+cpu \
   -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
If you are using a GPU, make sure to install the correct version of PyTorch that matches your CUDA version. You can find the appropriate command for your system on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Dataset Preparation

Download the [isic2020_challenge](link) classification dataset and structure the data as follows:
```
/path/to/isic2020_challenge/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

