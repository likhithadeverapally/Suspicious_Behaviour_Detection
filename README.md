# Suspicious_Behaviour_Detection

### 1. *Imports and Initial Setup*
   python
   import os
   import cv2
   import pafy
   import math
   import random
   import numpy as np
   import datetime as dt
   import tensorflow as tf
   from collections import deque
   import matplotlib.pyplot as plt

   from moviepy.editor import *
   from sklearn.model_selection import train_test_split
   from tensorflow.keras.layers import *
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.utils import to_categorical
   from tensorflow.keras.callbacks import EarlyStopping
   

   - *Libraries*:
     - cv2: OpenCV for image and video processing.
     - pafy: To fetch and handle YouTube video metadata (if used for online video sources).
     - math & random: Common Python libraries for math operations and randomness.
     - datetime as dt: Date and time utilities.
     - tensorflow & keras: Deep learning library used for building and training neural network models.
     - moviepy.editor: Video editing library, used to create video clips.
     - sklearn.model_selection.train_test_split: For splitting data into training and testing sets.
     - matplotlib.pyplot: For visualization and plotting.

   - *Random Seed Setup*:
     python
     seed_constant = 5
     np.random.seed(seed_constant)
     random.seed(seed_constant)
     tf.random.set_seed(seed_constant)
     
     - Ensures reproducibility by setting fixed seeds across numpy, random, and tensorflow.

### 2. *Dataset Path Configuration and Initial Data Visualization*
   python
   base_directory = '/kaggle/input/dataset'
   
   - Defines the dataset path containing "fights" and "noFights" folders.

   python
   for counter, selected_class_Name in enumerate(random_classes, 1):
       # Loop through random samples, read, and display frames with class names
   
   - Extracts frames from randomly chosen videos in each class, converts them from BGR to RGB format, and displays them with annotations.

### 3. *Frame Extraction Function*
   python
   def frames_extraction(video_path):
       # Processes each frame, resizing and normalizing them
   
   - *Function Purpose*: This function reads each video, extracts frames at intervals, resizes them to a fixed dimension, normalizes the frames, and stores them in a list.
   - *Key Steps*:
     - video_reader = cv2.VideoCapture(video_path): Opens the video file.
     - skip_frames_window: Determines the frame interval for sampling.
     - cv2.resize: Resizes frames to IMAGE_HEIGHT x IMAGE_WIDTH.
     - normalized_frame = resized_frame / 255: Normalizes pixel values between 0-1.

### 4. *Dataset Creation*
   python
   def create_dataset():
       # Loops through classes, creates labeled datasets, and returns numpy arrays
   
   - *Function Purpose*: This function iterates through each video in the classes directory, extracts frames using frames_extraction, and constructs feature and label arrays.
   - Converts features and labels into numpy arrays for model compatibility.

### 5. *Data Preparation*
   python
   features_train, features_test, labels_train, labels_test = train_test_split(...)
   
   - *train_test_split*: Divides data into 75% training and 25% testing sets.
   - *One-Hot Encoding*: to_categorical(labels) converts class labels into one-hot encoded vectors, suitable for categorical classification.

### 6. *LRCN Model Creation*
   python
   def create_LRCN_model():
       # Constructs the LRCN model architecture
   
   - *Function Purpose*: This function builds an LRCN model (Long-term Recurrent Convolutional Network), which combines Convolutional and LSTM layers to handle sequential data such as videos.
   - *Model Layers*:
     - *TimeDistributed Conv2D*: Applies convolution across frames to learn spatial features.
     - *MaxPooling2D*: Reduces dimensionality.
     - *LSTM*: Learns temporal dependencies across sequences.
     - *Dense Softmax Layer*: Output layer for classification.

### 7. *Model Training and Early Stopping*
   python
   model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])
   
   - *EarlyStopping*: Monitors validation accuracy and stops training if the accuracy does not improve within 10 epochs, preventing overfitting.
   - *model.fit*: Trains the model on the training data with a batch size of 4.

### 8. *Plotting Function for Training Metrics*
   python
   def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
       # Plots training/validation accuracy and loss for monitoring
   

   - *Purpose*: Plots metrics over epochs to observe model performance during training.

### 9. *Model Evaluation on Test Set*
   python
   acc = 0
   for i in range(len(features_test)):
       # Compares predictions with actual labels to compute accuracy
   
   - *Accuracy Calculation*: Loops over test samples, predicts each class, and compares it to the actual label.

### 10. *Single Video Prediction*
   python
   def predict_single_action(video_file_path, SEQUENCE_LENGTH):
       # Runs prediction on a single video file and outputs the predicted class
   
   - *Purpose*: Predicts the action being performed in a single video file.
   - *Steps*:
     - Processes the video, extracting and normalizing frames.
     - Predicts class probabilities for each frame sequence and prints the class with the highest probability.

### 11. *Predicting and Annotating Actions in Videos*
   python
   def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
       # Performs action recognition and saves video with predicted action labels
   
   - *Function Purpose*: This function annotates each frame of a video with predicted actions and saves it.
   - *Key Steps*:
     - Uses deque to store a fixed-length sequence of frames.
     - Reads, resizes, and normalizes frames in real-time.
     - Predicts and annotates the action on the video output file.

Each part of this code builds a full pipeline, from reading and processing video files to training and testing the LRCN model for detecting suspicious behavior in videos.
