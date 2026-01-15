# Comprehensive Hybrid CNN-XGBoost Architecture Documentation

## Overview
This project utilizes a comprehensive hybrid CNN-XGBoost architecture designed for effective PCB defect detection. The model leverages the strengths of convolutional neural networks (CNNs) for feature extraction and XGBoost for classification, providing high accuracy and robustness in detecting defects in PCB designs.

## Dataset Description
The dataset consists of labeled images of printed circuit boards (PCBs) with various types of defects. Each image is classified into categories representing the type of defect present, allowing the model to learn from both good and defective samples.

## Dual Architecture Approaches
### Convolutional Neural Network (CNN)
The CNN architecture is designed to extract intricate features from the input images. It includes layers such as convolutional layers, pooling layers, and dropout layers to enhance the model's learning capabilities and prevent overfitting.

### XGBoost
XGBoost is used as the classifier that takes the features produced by the CNN architecture and makes predictions. Its gradient boosting framework ensures that the model efficiently learns from the weak predictors to improve accuracy.

## Grad-CAM Explainability
Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the important regions in the input images that contribute to the model's predictions. This enhances interpretability and allows users to understand which parts of the PCB are influencing the predictions.

## Project Structure
```
pcb-defect-detection/
├── data/                  # Contains datasets
├── src/                   # Source code
│   ├── cnn.py            # CNN model definition
│   ├── xgboost.py        # XGBoost model definition
│   └── utils.py          # Utility functions
├── README.md              # Project documentation
└── requirements.txt      # Dependencies
```

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/thetushardudeja1/pcb-defect-detection.git
   cd pcb-defect-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples
To train the model, run:
```bash
python src/train.py
```
To evaluate the model, use:
```bash
python src/evaluate.py
```

## Technologies Used
- Python
- TensorFlow/Keras
- XGBoost
- OpenCV
- Matplotlib
- Scikit-learn

## Results
The hybrid architecture demonstrated an accuracy of over 95% on the validation set, significantly outperforming traditional methods in defect detection accuracy and speed.

## Author Information
This project is maintained by [thetushardudeja1](https://github.com/thetushardudeja1). For questions or suggestions, please feel free to open an issue or contact the author directly.