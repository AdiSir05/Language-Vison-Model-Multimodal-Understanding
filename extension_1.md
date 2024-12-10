# Extension 1

This repository contains a Jupyter Notebook implementing an ViT to classify images as watermarked/not watermarked.

---

## Setup Instructions

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision matplotlib tqdm
```

### 2. Prepare the Dataset

You can download the zip of the dataset at: https://drive.google.com/file/d/1pXP8HH4EW9l3JThleNr05d8jz96jEoIJ/view?usp=sharing

Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as `extension_1.py`.


### 4. Include Watermarking Model

You can download the zip of the models at: https://drive.google.com/file/d/1Jh4WotI3U0hwWDw96xgg93DRo2OS2J-F/view?usp=sharing

Unzip the pretrained watermarking and ViT model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `extension_1.py`
Make sure that the files: `stegastamp_wm.py`, `VisionTransformer.py` and `wm_stegastamp_encoder.pth` are in the `models` directory.

### 5. Run the script
Launch the python file:
```bash
python extension_1.py
```

---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Loads images and captions.
    - Produces watermarked images for the dataset
2. **Model**
    - We load the model defined in VisionTransformer.py and train it to classify watermarked vs not watermarked images.
3. **Model Evaluation**:
    - Compute metrics F1, Precision, Recall to evaluate classification of watermarked images.
4. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.