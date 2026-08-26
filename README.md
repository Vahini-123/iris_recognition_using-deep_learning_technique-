# Iris Recognition Using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Neural%20Network-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Data%20Processing-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square" alt="Matplotlib">
</p>

<p align="center">
  <strong>A deep learning-based biometric system for iris recognition and person identification.</strong>
</p>

---

## 📌 Overview

**Iris Recognition Using Deep Learning** is a biometric authentication system developed using **Deep Learning and Computer Vision** techniques.

The system processes eye images, extracts iris-related features, and uses a **Convolutional Neural Network (CNN)** to identify individuals based on their unique iris patterns.

The project also provides a **Tkinter GUI** for interacting with the recognition system and visualizing model training performance.

---

## ✨ Features

- 👁️ Iris image preprocessing
- 🔍 Iris feature extraction
- 🧠 CNN-based deep learning model
- 🔐 Biometric person identification
- 🖥️ Tkinter graphical user interface
- 📊 Accuracy and loss visualization
- 💾 Trained model saving and loading
- 🎯 Prediction of person identity from test images

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| OpenCV | Image processing |
| NumPy | Numerical computation |
| Matplotlib | Graph visualization |
| Scikit-image | Image processing |
| Tkinter | GUI development |

---

## 🔄 Project Workflow

    Iris Dataset
          ↓
    Image Preprocessing
          ↓
    Iris Feature Extraction
          ↓
    CNN Model Training
          ↓
    Save Trained Model
          ↓
    Upload Test Iris Image
          ↓
    Feature Processing
          ↓
    CNN Prediction
          ↓
    Person Identification

---

## 🧠 CNN Architecture

The project uses a Convolutional Neural Network with convolution, pooling, flattening, and dense layers.

    Sequential Model
          ↓
    Conv2D (32 filters, 3×3)
          ↓
    MaxPooling2D
          ↓
    Flatten
          ↓
    Dense (256 neurons)
          ↓
    Dense (108 classes)
          ↓
    Softmax Prediction

Example model structure:

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=108, activation='softmax'))

---

## 🚀 Installation

Install the required Python libraries:

    pip install tensorflow keras opencv-python matplotlib scikit-image numpy

---

## ▶️ How to Run

Clone the repository:

    git clone https://github.com/Vahini-123/iris_recognition_using-deep_learning_technique-.git

Navigate to the project directory:

    cd iris_recognition_using-deep_learning_technique-

Run the main application:

    python Main.py

---

## 📊 Dataset

The project uses iris image data for biometric recognition and person identification.

The dataset is processed before being provided to the CNN model for training and prediction.

---

## 🖼️ Screenshots

### Main GUI

![Main GUI](screenshots/gui.png)

### Model Training

![Model Training](screenshots/training.png)

### Prediction Result

![Prediction Result](screenshots/result.png)

### Accuracy & Loss Graph

![Accuracy and Loss Graph](screenshots/graph.png)

---

## 💡 Applications

- 🔐 Biometric authentication
- 🛡️ Security systems
- 👤 Identity verification
- 🚪 Access control systems
- 🏢 Secure facility authentication

---

## 🔮 Future Enhancements

- Real-time iris recognition
- Webcam integration
- Improved CNN architecture
- Higher recognition accuracy
- Cloud deployment
- Real-time biometric authentication
- Larger and more diverse datasets

---

## 👩‍💻 Author

**Vennapusa Vahini**

Computer Science & Engineering (Data Science) Graduate

[GitHub](https://github.com/Vahini-123)

---

<p align="center">
  Built with Python, TensorFlow, Keras, and OpenCV.
</p>
