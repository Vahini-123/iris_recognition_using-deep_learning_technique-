# Iris Recognition using Deep Learning Technique

## Overview
This project is a biometric authentication system developed using Deep Learning and Computer Vision techniques. The system identifies individuals based on unique iris patterns from eye images.

---

## Features
- Iris image preprocessing
- Iris feature extraction using OpenCV
- CNN-based Deep Learning model
- Person identification using iris recognition
- GUI interface using Tkinter
- Accuracy and loss graph visualization

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- Scikit-image

---

## Project Workflow

1. Upload iris dataset
2. Extract iris features
3. Train CNN model
4. Save trained model
5. Upload test iris image
6. Predict person ID
7. Display prediction results

---

## CNN Architecture

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=108, activation='softmax'))
```

---

## Installation

Install required libraries:

```bash
pip install tensorflow keras opencv-python matplotlib scikit-image numpy
```

---

## How to Run

```bash
cd IrisRecognition
python Main.py
```

---

## Dataset
The project uses iris image datasets for biometric recognition and identification.

---

## Screenshots

### Main GUI
![GUI Screenshot](screenshots/gui.png)

### Model Training
![Training Screenshot](screenshots/training.png)

### Prediction Result
![Prediction Screenshot](screenshots/result.png)

### Accuracy & Loss Graph
![Graph Screenshot](screenshots/graph.png)

---

## Applications
- Biometric authentication
- Security systems
- Identity verification
- Access control systems

---

## Future Enhancements
- Real-time iris recognition
- Webcam integration
- Cloud deployment
- Improved CNN accuracy

---

## Author
Vahini Vennapusa

---

## GitHub Repository
https://github.com/Vahini-123/iris_recognition_using-deep_learning_technique-
