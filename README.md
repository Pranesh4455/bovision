# 🐄 BoVision: AI-Powered Image Classification System

BoVision is a deep learning-based image classification system designed to distinguish between **cows and buffaloes** using transfer learning. The project leverages MobileNetV2 and a carefully designed preprocessing pipeline to achieve high accuracy on a relatively small dataset.

## 🚀 Project Overview

In livestock management, distinguishing between cows and buffaloes is often done manually and can be error-prone. BoVision automates this process using computer vision, making it useful for:
- Farm management systems  
- Livestock insurance verification  
- Veterinary diagnostics  
- Agricultural data collection  

## 🧠 Key Features

- ✅ Binary classification: **Cow vs Buffalo**
- ✅ Transfer learning using **MobileNetV2**
- ✅ High accuracy: **97.59% validation accuracy**
- ✅ Lightweight model (~8.63 MB)
- ✅ Deployable on mobile and edge devices
- ✅ Built and trained entirely on Google Colab

## 📊 Model Performance

| Metric                 |  Value    |
|------------------------|-----------|
| Training Accuracy      | 97.27%    |
| Validation Accuracy    | 97.59%    |
| Epochs                 | 15        |
| Trainable Parameters   | 2,562     |


## ⚙️ Tech Stack

- **Framework:** TensorFlow / Keras  
- **Model:** MobileNetV2 (ImageNet pretrained)  
- **Environment:** Google Colab (GPU)  
- **Language:** Python  
- **Data Pipeline:** tf.data API  

