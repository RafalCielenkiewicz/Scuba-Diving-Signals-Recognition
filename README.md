# Real-Time Scuba Diving Gesture Recognition System

## Project Overview

This project aims to develop a system capable of recognizing scuba diving hand signals in real-time. The system utilizes YOLO for hand detection and a custom Convolutional Neural Network (CNN) for gesture classification. A self-collected dataset was used to train the models, ensuring high accuracy in real-world scenarios.

## Key Features

*   **Real-time hand detection:** Utilizes YOLO to accurately detect hands in video streams.
*   **Gesture classification:** Employs a custom CNN to classify detected hand gestures.
*   **Custom dataset:** A dataset of 1080 images per gesture, collected to train the models.
*   **Data augmentation:** Techniques like rotation, zoom, and flipping were used to improve model robustness.
*   **Real-time output display:** Recognized gestures and confidence levels are displayed on the video feed.

## Technologies Used

*   **Frameworks/Libraries:** TensorFlow/Keras, PyTorch, OpenCV

## Results

The YOLO-based hand detection model demonstrated high accuracy and real-time performance. The CNN-based gesture recognition model also achieved promising results, especially after applying data augmentation techniques.

### Examples:

* **Hand Detection:** YOLO effectively detects hands under various conditions.
    <img src="https://github.com/user-attachments/assets/636adc52-be1e-4b36-baee-fa93209a1f10" width="1000" alt="Hand Detection Example">

* **Gesture Recognition:** The CNN model accurately classifies gestures.
    <img src="https://github.com/user-attachments/assets/34e177b9-c6cd-4f79-90fe-59be92e34c37" width="1000" alt="Gesture Recognition Example 1">
    <img src="https://github.com/user-attachments/assets/48211c8a-f321-4448-afc0-e022f421dc9f" width="1000" alt="Gesture Recognition Example 2">
    <img src="https://github.com/user-attachments/assets/abc73226-e448-4d6e-969a-72b27fc9f285" width="1000" alt="Gesture Recognition Example 3">

