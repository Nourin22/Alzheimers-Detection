Alzheimer’s Disease Detection Using Deep Learning

Project Overview:

This project implements a deep learning model to detect Alzheimer’s disease from MRI scan images. Using Convolutional Neural Networks (CNNs), the model classifies MRI images into four categories based on dementia severity:

Mild Demented
Moderate Demented
Non Demented
Very Mild Demented

Early detection of Alzheimer's is crucial for timely medical intervention. Our model provides an automated classification system with high accuracy, making it a potential tool for assisting medical professionals.

Dataset Details:
Source: Kaggle - Alzheimer’s MRI Disease Classification Dataset

Class Distribution:
Some classes have fewer samples, requiring data augmentation to balance the dataset.

Preprocessing & Data Augmentation:
Preprocessing:
✅ Resizing: Standardizing images to a fixed size
✅ Normalization: Scaling pixel values between 0 and 1

Data Augmentation (Applied to Minority Classes):

✅ rotation_range=20 → Randomly rotates images
✅ width_shift_range=0.2, height_shift_range=0.2 → Shifts images along axes
✅ shear_range=0.2 → Applies shearing transformations
✅ zoom_range=0.2 → Random zooming
✅ horizontal_flip=True → Flips images horizontally
✅ fill_mode='nearest' → Fills missing pixels after transformation

Deep Learning Model:

The Convolutional Neural Network (CNN) is designed with the following architecture:

Convolutional Layers: Extract features from MRI scans
Max-Pooling Layers: Reduce dimensionality and retain key patterns
Fully Connected Layers: Flatten and process extracted features
Softmax Activation: Classifies images into four categories

Implementation & Evaluation:

Metrics: Accuracy, Precision, Recall
Final Accuracy Achieved: 98.7% 🎯

Deployment & Testing:

To make the model accessible for real-world testing, we built a Streamlit Web App for easy interaction. Users can upload MRI images, and the model will classify them in real time.

