
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
Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as your Jupyter notebook.


### 4. Include Watermarking Model

Unzip the pretrained watermarking model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your Jupyter notebook.

### 5. Run the Notebook
Launch the Jupyter Notebook:
```bash
jupyter notebook simple_baseline.ipynb
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
