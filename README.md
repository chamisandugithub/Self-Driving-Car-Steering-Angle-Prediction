#  Self-Driving Car Steering Angle Prediction

A deep learning project for predicting steering angles in autonomous driving using a CNN model with **Weighted MSE Loss** and performance analysis.

##  Project Overview
This project trains a CNN model to predict steering angles from road images and evaluates prediction quality across different turning ranges.

##  Features
- CNN-based steering angle prediction
- Custom **Weighted MSE Loss** for near-zero steering accuracy
- Data augmentation (flip + brightness adjustment)
- Performance evaluation with RMSE & direction accuracy
- Error analysis by steering angle bins
- Visual analytics using Matplotlib & Seaborn

##  Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

##  Model Workflow
1. Load balanced driving dataset  
2. Preprocess road images  
3. Train CNN model  
4. Apply weighted loss function  
5. Evaluate predictions  
6. Generate visual performance reports  


Improve autonomous driving steering predictions by reducing errors, especially in **near-zero steering situations** where precision is critical.
