# ECE5831-Fall-2024-Final-Project

## Project Title

**Deepfake Video Detection: Leveraging Machine Learning Techniques for Media Authenticity**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Proposal](#project-proposal)
3. [Progress Updates](#progress-updates)
4. [Final Report](#final-report)
5. [Presentation](#presentation)
6. [Dataset](#dataset)
7. [Demo](#demo)
8. [How to Run](#how-to-run)

---

## Introduction

**Course:** ECE 5831 - Pattern Recognition and Neural Networks  
**Instructor:** Professor Jaerock Kwon  
**Group Members:**  
- Aishwarya Dekhane ([adekhane@umich.edu](mailto:adekhane@umich.edu))  
- Shubhang Vangari ([shubbu@umich.edu](mailto:shubbu@umich.edu))  
- Atharva Pore ([atharva@umich.edu](mailto:atharva@umich.edu))  

This project focuses on detecting deepfake videos using advanced machine learning techniques such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). Additional methods like optical flow and facial landmark detection are leveraged to identify inconsistencies, ensuring robust detection of manipulated media.

---

## Project Proposal

### Abstract
This project aims to combat the misuse of deepfake technology by developing a robust detection system. Techniques such as CNNs (Inception, VGG16), RNNs, optical flow analysis, and facial landmark detection are used to detect spatial and temporal inconsistencies in videos. The system focuses on enhancing media authenticity and public trust.

### Motivation
Deepfake technology poses significant ethical, legal, and societal risks. Detecting manipulated videos is crucial to combat misinformation, identity fraud, and privacy breaches.

### Prior Work
- CNN architectures (ResNet50, VGG16, MesoNet) for spatial analysis.
- RNNs and LSTMs for temporal inconsistencies.
- Advanced techniques like attention mechanisms, transformers, and optical flow.

### Goals
- Develop a detection pipeline combining spatial and temporal analysis.
- Integrate advanced methods like optical flow and facial landmark detection.
- Achieve high accuracy on the DFDC dataset.

### Deliverables
- A complete detection system (codebase).
- Presentation video summarizing project outcomes.
- A final IEEE-format project report.

---

## Progress Updates

### Progress Update 1  

#### Achievements:
- Data preprocessing using the DFDC dataset.
- Implemented CNN architectures (ResNet50, VGG16).
- Initial model training achieving 85% accuracy.

#### Challenges:
- Handling dataset imbalance.
- Computational constraints with large datasets.

### Progress Update 2 

#### Achievements:
- Integrated optical flow and facial landmark detection.
- Improved accuracy by combining CNN outputs with advanced techniques.

#### Challenges:
- Addressing high-quality deepfake detection.
- Reducing false positives.

---

## Final Report

Report: [Link to the report](#).

---

## Presentation

Presentation Video: [https://youtu.be/3BpoVp-kmeo](https://youtu.be/3BpoVp-kmeo)  
PPT Presentation: [https://drive.google.com/file/d/1Rtn7cRN9cfk6QgUW7tgGLlYS9yh_s1sU/view?usp=drive_link](https://drive.google.com/file/d/1Rtn7cRN9cfk6QgUW7tgGLlYS9yh_s1sU/view?usp=drive_link)

---

## Dataset

**Dataset Description:**  
- **Source:** Deepfake Detection Challenge (DFDC) dataset.
- **Details:** Includes real and fake videos for training deepfake detection models.

[https://drive.google.com/drive/folders/13CPUvGrRanP3hVPrWt_BIW2O5j6cAMVF?usp=drive_link](https://drive.google.com/drive/folders/13CPUvGrRanP3hVPrWt_BIW2O5j6cAMVF?usp=drive_link)

---

## Demo

**Demo Video:** [https://youtu.be/pnoy0iG8kOU](https://youtu.be/pnoy0iG8kOU)

---

## Models

**Model Folder Link:** [https://drive.google.com/drive/folders/1zsC9WazG16_BLv_-W5IcgfAYOMMxMwfP?usp=drive_link](https://drive.google.com/drive/folders/1zsC9WazG16_BLv_-W5IcgfAYOMMxMwfP?usp=drive_link)

---

## How to Run

The project code is included in `final-project.ipynb`.

### Requirements
- Python 3.8+
- Libraries: TensorFlow, Keras, OpenCV, NumPy, pandas, matplotlib.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/ece5831-2024-final-project.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook final-project.ipynb
   ```

### Execution
1. Follow the steps outlined in `final-project.ipynb` to reproduce the results.
2. Modify parameters in the notebook for experimentation.

---

## Acknowledgments

We extend our gratitude to:
- Our professor for their invaluable guidance and support.
- The creators of the DFDC dataset for providing a rich resource.
- University of Michigan, Dearborn, for access to facilities and resources.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or feedback, contact:
- Aishwarya Dekhane ([adekhane@umich.edu](mailto:adekhane@umich.edu))
- Shubhang Vangari ([shubbu@umich.edu](mailto:shubbu@umich.edu))
- Atharva Pore ([atharva@umich.edu](mailto:atharva@umich.edu))
