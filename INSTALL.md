# Installation

We provide installation instructions.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Run:
```
pip install -r requirements.txt
```

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

