
# Extention 1

This repository contains a Jupyter Notebook and associated Python scripts implementing a watermark detection baseline using a Vision Transformer (ViT) classifier. The system leverages Vision Transformer Classifier to classify images as either watermarked or non-watermarked. It employs metrics such as F1, Precision, Recall, and accuracy to evaluate model performance.

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

Make sure that the `data` directory is in the same directory as `extention.py`.


### 4. Include Watermarking Model

Unzip the Vision Transformer and pretrained watermarking model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `extention.py`
Make sure that the files: `stegastamp_wm.py`, `wm_stegastamp_decoder.pth`, `wm_stegastamp_encoder.pth` ,and `VisionTranformer.py` is in the `models` directory.

### 5. Run the script
Launch the python file:
```bash
python extention.py
```

---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Utilizes the COCO Captions dataset for training and validation.
    - Applies transformations such as resizing, center cropping, and tensor conversion.
    - Integrates the StegaStamp watermark encoder to embed signatures into images.
    - Constructs a custom dataset (CocoCaptionWMDataset) that includes both watermarked and non-watermarked images with corresponding labels.
2. **Model**
    - Vision Transformer Classifier
        - Defined in VisionTransformer.py, this model implements a Vision Transformer architecture tailored for image classification.
    - Watermark Encoder
        - StegaStampEncoder encodes a binary signature into an image to create a watermarked version.
        - The encoder is loaded with pretrained weights from the models directory.
3. **Model Evaluation**:
    - Compute metrics F1, Precision, Recall, and Accuracy to evaluate classification of watermarked images.
3. **Training Process**:
    - Utilizes the AdamW optimizer with a learning rate of 1e-4.
    - Employs Cross Entropy Loss for classification.
    - Trains the model over 10 epochs, tracking and visualizing the loss after each epoch.
5. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.