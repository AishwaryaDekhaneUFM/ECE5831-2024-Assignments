# HW10: Dogs vs Cats Classification

This repository contains the implementation for HW10, a project to classify images of dogs and cats using a neural network in TensorFlow. Below is an overview of the implementation details, requirements, and usage instructions.

---

## Table of Contents
- [Overview](#overview)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Implementation](#implementation)
- [How to Use](#how-to-use)
- [Results](#results)
- [Files](#files)
- [Links](#links)

---

## Overview

The project implements a `DogsCats` class to process the "Dogs vs Cats" dataset and build a convolutional neural network (CNN) for classification. Key features include:
- Dataset preparation: Splitting the data into training, validation, and testing sets.
- Neural network design: Using convolutional and pooling layers for improved performance.
- Training and evaluation: Plotting metrics like accuracy and loss.

---

## Dataset Preparation

1. Download the dataset from Kaggle: [Dogs vs Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data).  
2. Unzip the file and rename the directory to `dogs-vs-cats-original`.
3. The data is split as follows:
   - **Training**: 2,400 - 11,999
   - **Validation**: 0 - 2,399
   - **Testing**: 12,000 - 12,499

---

## Model Architecture

The neural network comprises:
- Convolutional layers for feature extraction.
- Pooling layers to reduce dimensionality.
- Dense layers for classification.
- Additional data augmentation for improved generalization.

---

## Implementation

The `DogsCats` class provides the following methods:
- **`make_dataset_folders()`**: Prepares dataset folders for train, validation, and test splits.
- **`_make_dataset()`**: Creates TensorFlow datasets using `image_dataset_from_directory()`.
- **`make_dataset()`**: Initializes `train_dataset`, `valid_dataset`, and `test_dataset`.
- **`build_network()`**: Constructs the neural network and compiles it.
- **`train()`**: Trains the model and saves it.
- **`load_model()`**: Loads a pre-trained model.
- **`predict()`**: Makes predictions on a given image file.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Prepare the dataset:
   - Place the unzipped dataset in the directory `dogs-vs-cats-original`.
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook module10.ipynb
   ```
4. Follow the instructions in the notebook to train the model, evaluate it, and make predictions.

---

## Results

The trained model achieves an accuracy of over 70% on the validation set. Detailed training and validation metrics (accuracy and loss) are plotted in the notebook.

---

## Files

- **`dogs_cats.py`**: Contains the `DogsCats` class implementation.
- **`module10.ipynb`**: Jupyter notebook for running the implementation and showcasing results.
- **Dataset Directory**: Ensure the dataset directory is structured as:
  ```
  dogs-vs-cats-original/
      train/
          dog/
          cat/
  ```

---

## Links

- Dataset: [Kaggle - Dogs vs Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)  
- Model File: [Download .keras Model](https://drive.google.com/file/d/1BxHqtpflurJbH4Rgr8E1Yz4-vWd7MZBK/view?usp=drive_link)

---

## Prepared By

- Aishwarya Dekhane - adekhane@umich.edu
