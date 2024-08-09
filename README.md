# Eye-Tracker
This project was developed as part of a diploma project for the Belarussian State University of Informatics and Radioelectronics. The main idea was to explore various ways of finding eyes and tracking eye gaze on a video from a webcam and implement one of them. In this solution, MediaPipe is used to detect eyes in the first frame. Then a convolutional neural network (CNN) analyzes the eye region on the following frames, resulting in enhanced accuracy and speed.
# Functionality
## Settings tab
- Chousing model's version.
- Downloading a model.
- Chousing camera.
## Calibration tab
Calibration is a necessary step for cheating detection (tracking whether the user is looking at the screen).
## Gaze Direction tab
Finding eyes on a video from a webcam and tracking eye gaze.
## Cheating Detection tab
Detects whether the user is looking at the screen or not.
# Installation 
- Download archive.
- Unpack archive.
- Run the command ```pip install -r requirements.txt```.
- Open the project folder in the terminal.
- Run the command ```python EyeTracker.py```.

Another way is cloning the project, installing all required libraries from requirements.txt and running EyeTracker.py in the IDE.
