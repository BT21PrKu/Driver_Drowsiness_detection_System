# Driver_Drowsiness_detection_System

## Introduction

The goal of this project is to develop a system that can detect driver drowsiness in real-time using computer vision techniques.

### Motivation
Drowsy driving is a major cause of road accidents. Detecting drowsiness early can help prevent accidents and save lives.

### Technologies Used
- Python
- OpenCV
- dlib
- imutils

### Hardware
The system was tested using a webcam to capture video frames of the driver's face.

## System Workflow

### 1. Face Detection and Landmark Identification
- Utilized dlib's pre-trained face detector to locate the driver's face in each frame.
- Employed a shape predictor to identify 68 facial landmarks - "shape_predictor_68_face_landmarks.dat".

### 2. Feature Extraction
- Calculated key features such as the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) from the facial landmarks to determine signs of drowsiness.

### 3. Drowsiness Detection
- Set threshold values for EAR and MAR to detect drowsiness and yawning.
- Flagged the driver as drowsy if the EAR fell below a certain threshold for consecutive frames.
- Indicated yawning if a high MAR was detected.

### Real-Time Processing
- The video stream was processed in real-time, and visual alerts were displayed on the screen if drowsiness or yawning was detected.

## Evaluation and Results

### Ground Truth Data
Collected ground truth labels by manually annotating video frames to indicate whether the driver was drowsy or not.

## Conclusion

This project demonstrates a practical application of computer vision in enhancing road safety. By detecting drowsiness in real-time, the system can alert drivers and potentially reduce the number of accidents caused by drowsy driving.
