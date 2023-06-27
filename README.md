# Carvana U-Net Segmentation

## 1. Project Summary

In this repo, we implement U-Net Semantic Segmentation from scratch for Carvana Image Masking Challenge. The main difference between this architecture and the original paper is that we use padded convolutions instead of valid (unpadded) convolutions.

### 1.1. Architecture

The U-Net architecture is a popular CNN model for image segmentation. It was first introduced in 2015, and has since been widely adopted in various field. The U-Net consists of an encoder path to capture features, a decoder path for generating a segmentation map, and skip connections connect the encoder and decoder paths, enabling the model to combine low-level and high-level features. The U-Net effectively captures details and context, making it ideal for segmentation tasks. The U-Net architecture is as follow.

### 1.2. Dataset

### 1.3. Results

## 2. Usage

### 2.1. Dependencies

### 2.2. Training
We use automatic mixed precision to speed up the training process. The model is trained for 10 epochs. To perform training, you could run:

python train.py

### 2.3. Testing


## 2.4. Speed Up Inference
