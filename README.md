Drowsiness Detection System
Overview
The Drowsiness Detection System is a real-time safety technology designed to prevent accidents caused by driver fatigue. By leveraging computer vision and machine learning techniques, the system monitors facial landmarks, eye closure, and other indicators of drowsiness to alert the driver when signs of fatigue are detected. This project aims to enhance road safety and reduce the risk of accidents caused by drowsy driving.

Features
Real-Time Monitoring: Utilizes a webcam to capture live video streams for continuous monitoring.

Eye Aspect Ratio (EAR): Calculates the EAR to detect prolonged eye closure.

Mouth Open Ratio (MOR): Monitors yawning as an indicator of drowsiness.

Nose-to-Lip Ratio (NLR): Tracks facial landmarks for additional accuracy in detecting fatigue.

Alert System: Triggers an audible alarm when drowsiness is detected.

Non-Intrusive Design: Ensures driver comfort while monitoring in real time.

Technologies Used
Software
Python Libraries:

OpenCV: For image and video processing.

dlib: For facial landmark detection.

pygame: For playing alert sounds.

scipy: For calculating Euclidean distances in facial landmark analysis.

imutils: For image resizing and preprocessing.

Hardware
A standard webcam for capturing live video streams.

How It Works
Facial Landmark Detection:

Detects the face and extracts key regions such as eyes, mouth, and nose using a pre-trained model (shape_predictor_68_face_landmarks.dat).

Feature Ratios:

Calculates the following ratios:

Eye Aspect Ratio (EAR): Detects prolonged eye closure.

Mouth Open Ratio (MOR): Identifies yawning.

Nose-to-Lip Ratio (NLR): Tracks changes in facial posture.

Threshold Monitoring:

If EAR falls below a certain threshold or MOR/NLR exceeds predefined limits for a specific duration, the system flags drowsiness.

Alert Mechanism:

Plays an audible alarm to alert the driver when drowsiness is detected.

Installation
Clone the repository:

bash
git clone https://github.com/your_username/Drowsiness_Detection_System.git
cd Drowsiness_Detection_System

Download the pre-trained model for facial landmark detection:
Place shape_predictor_68_face_landmarks.dat in the models/ directory.

Run the application:
bash
python drowsiness_detection.py

Usage
Ensure your webcam is connected and functional.
Run the script to start real-time monitoring.

Press q to exit the application.
