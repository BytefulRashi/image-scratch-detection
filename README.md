# **Scratch Detection and Classification**

**Scratch Detection and Classification using Mask R-CNN and ResNet50**

This project leverages deep learning techniques to detect scratches on surfaces, classify images into "Good" or "Bad" categories, and localize scratches using bounding boxes and masks. It combines the power of ResNet50 for classification and Mask R-CNN for instance segmentation.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
    - [Classification Using ResNet50](#classification-using-resnet50)
    - [Scratch Localization Using Mask R-CNN](#scratch-localization-using-mask-r-cnn)
5. [Results](#results)
6. [Usage](#usage)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Future Enhancements](#future-enhancements)
9. [Acknowledgments](#acknowledgments)

---

## **Project Overview**

The goal of this project is to automate the detection and classification of scratches on surfaces. It has two main objectives:
1. **Image Classification:** Classify images as "Good" (no scratches) or "Bad" (scratches present) using ResNet50.
2. **Scratch Localization:** Identify and localize scratches on "Bad" images using Mask R-CNN, providing precise bounding boxes and masks for scratch regions.

---

## **Installation**

Follow these steps to set up the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/scratch-detection.git
   cd scratch-detection
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify TensorFlow and Keras Compatibility:**
   Ensure that TensorFlow version 2.6 or newer is installed:
   ```bash
   pip install tensorflow==2.6.0
   ```

4. **Install Mask R-CNN Library:**
   ```bash
   pip install git+https://github.com/matterport/Mask_RCNN.git
   ```

---

## **Dataset**

The dataset contains images with and without scratches. The dataset is divided into:
- **Good Images:** Images without scratches.
- **Bad Images:** Images with visible scratches.

For "Bad" images, labeled masks and bounding box annotations are provided for scratch regions. Ensure your dataset is organized as follows:
```
dataset/
│
├── train/
│   ├── images/
│   ├── masks/
│
├── validation/
│   ├── images/
│   ├── masks/
│
├── test/
    ├── images/
    ├── masks/
```

---

## **Methodology**

### **Classification Using ResNet50**
1. **Model Selection:** ResNet50 is pretrained on ImageNet and fine-tuned for binary classification.
2. **Training:** The model is trained to classify images as "Good" or "Bad" based on the presence of scratches.
3. **Fine-tuning:** Hyperparameters and layers are adjusted for optimal performance on the scratch dataset.

### **Scratch Localization Using Mask R-CNN**
1. **Instance Segmentation:** Mask R-CNN identifies scratches in images and creates pixel-level masks.
2. **Thresholding:** Scratch areas are measured, and a threshold is set to classify "Bad" images.
3. **Contour Detection and Bounding Boxes:** Contours are detected, and bounding boxes are drawn to visualize scratch locations.

---

## **Results**

### **Classification Results**
- **Precision (Bad):** 0.96  
- **Recall (Bad):** 0.78  
- **F1-Score (Bad):** 0.86  
- **Accuracy:** 95%  

### **Scratch Localization**
- **Localization Accuracy:** High precision in identifying scratch regions and bounding box placement.

---

## **Usage**

### **Run the Project**
1. **Train the Classification Model:**
   ```bash
   python train_resnet50.py
   ```

2. **Train the Scratch Localization Model:**
   ```bash
   python train_maskrcnn.py
   ```

3. **Test the Models:**
   ```bash
   python test_models.py
   ```

### **Predict on New Images**
Use the provided script to classify and localize scratches:
```bash
python predict.py --image_path /path/to/image.jpg
```

---

## **Evaluation Metrics**

1. **Classification Metrics:**
   - Precision, Recall, and F1-Score for both "Good" and "Bad" classes.
   - ROC Curve and AUC for model performance evaluation.

2. **Localization Metrics:**
   - Intersection Over Union (IoU) for bounding box and mask accuracy.

---

## **Future Enhancements**
- Incorporate multi-class classification for different types of scratches.
- Train the model on a more extensive and diverse dataset.
- Improve scratch detection on challenging backgrounds.
- Optimize the pipeline for real-time inference.

---

## **Acknowledgments**
This project is based on Mask R-CNN by Matterport and ResNet50 pretrained on ImageNet. Special thanks to the open-source community for their tools and libraries.

---
