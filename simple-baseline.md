
# Simple Baseline

This repository contains a Jupyter Notebook implementing a **naive** image-captioning and watermark captioning baseline using deep learning. It employs metrics such as F1, METEOR, BLEU, and ROUGE to evaluate model performance.

---

## Setup Instructions

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision transformers bert-score evaluate tqdm matplotlib numpy
```

### 2. Prepare the Dataset

You can download the zip of the dataset at: https://drive.google.com/file/d/1pXP8HH4EW9l3JThleNr05d8jz96jEoIJ/view?usp=sharing

Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as `simple_baseline.py`.


### 4. Include Watermarking Model

Unzip the pretrained watermarking model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `simple_baseline.py`
Make sure that the files: `stegastamp_wm.py`, `wm_stegastamp_decoder.pth`, and `wm_stegastamp_encoder.pth` is in the `models` directory.

### 5. Run the script
Run the python file:
```bash
python simple_baseline.py
```

---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Loads images and captions.
    - Produces watermarked images for the dataset
2. **Model**
    - For each image, finds the closest image (using L2 distance) and outputs in caption/classification
3. **Model Evaluation**:
    - Computes metrics (METEOR, BLEU, ROUGE) using the `evaluate` library to evaluate the produced captioning for images. Watermarked images are captioned with their signature.
    - Compute metrics F1, Precision, Recall to evaluate classification of watermarked images.
4. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.
