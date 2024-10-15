# Rock-Paper-Scissors AI Project

Welcome to the Rock-Paper-Scissors AI project! This project involves creating a neural network model using Teachable Machine and Python to recognize hand gestures (rock, paper, and scissors) both from static images and in real-time using a webcam.

## Project Overview

This project is divided into two main parts:
1. **Static Image Classification**: Classifying rock, paper, and scissors from an image file.
2. **Live Classification**: Using a webcam to classify rock, paper, and scissors gestures in real time.

## Deliverables

- `README.md`
- `rock-paper-scissors.py`
- `rock-paper-scissors-live.py`
- `teachable.ipynb`

## Requirements

Ensure you have the following installed:
- Python 3.6+
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You may need to create a new conda environment with specific settings to use the trained neural network model.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/rock-paper-scissors-ai.git
   cd rock-paper-scissors-ai
   ```

2. **Set up the Python environment**:
   ```sh
   conda create --name rps-env python=3.8
   conda activate rps-env
   pip install tensorflow opencv-python numpy matplotlib
   ```

3. **Download the trained model**:
   - Train your model on Teachable Machine for rock, paper, and scissors. Download the model and place it in the project directory.
   - Also, download the `labels.txt` file containing the class names.

## Scripts

### 1. Rock-Paper-Scissors from Image (`rock-paper-scissors.py`)

This script takes an image file as input, processes it, and predicts whether it is rock, paper, or scissors with a confidence score.

**Usage**:
```sh
python rock-paper-scissors.py path_to_image.jpg
```

### 2. Rock-Paper-Scissors Live (`rock-paper-scissors-live.py`)

This script uses your webcam to recognize rock, paper, and scissors gestures in real time. 

**Usage**:
```sh
python rock-paper-scissors-live.py
```

### 3. Model Training and Testing (`teachable.ipynb`)

This Jupyter notebook is used for training and testing the Rock-Paper-Scissors model using Teachable Machine data.

## Video Demonstration

Check out the full demonstration and tutorial on YouTube:

[Watch the video here!](https://www.youtube.com/your-video-link)

## Conclusion

This project offers a hands-on approach to integrating machine learning with real-time applications, making it a fun and interactive way to learn about AI, Python, and computer vision.


Feel free to reach out if you have any questions or need further assistance!
