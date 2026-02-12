 # Brain Tumor Detection using Transfer Learning (MobileNetV2)
 ## Project Overview

This project is a Deep Learning-based medical image classification system that detects the presence of brain tumors from MRI images.

The model uses MobileNetV2 (pretrained on ImageNet) with transfer learning and is deployed as a live web application on Hugging Face.

 ## Live Demo

### Try the app here:
(https://huggingface.co/spaces/ben-ds/brain_tumour_detection)

## Tech Stack

Python

TensorFlow / Keras

MobileNetV2 (Transfer Learning)

OpenCV

Gradio 

Hugging Face Spaces (Deployment)

## Model Details

Base Model: MobileNetV2

Input Shape: (224, 224, 3)

Pretrained on: ImageNet

Transfer Learning: Yes

Optimizer: Adam 

Loss Function: Binary CategoricalCrossentropy 

Accuracy Achieved:
    Train accuracy:95
    Vaidation accuracy:91
    Test accuracy :86
    
## Workflow

Data preprocessing and image resizing

Loading MobileNetV2 base model

Freezing base layers

Adding custom classification head

Model training and evaluation

Saving trained model

Deployment using Hugging Face
