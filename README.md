# PCB Defect Detection using Deep Learning

A research-grade deep learning pipeline for automated multi-class defect classification on printed circuit boards (PCBs) using patch-based analysis and convolutional neural networks.

## Overview

This project implements a CNN-based classification system for detecting and classifying manufacturing defects in PCBs. The system processes digitized PCB images at the patch level, enabling localized defect identification and providing explainability through gradient-based attention visualization. This approach is essential for quality assurance in electronics manufacturing and bridges computer vision research with industrial inspection applications.

### Problem Statement
Modern PCB manufacturing requires automated quality control to detect multiple defect types with high accuracy and speed. Manual inspection is labor-intensive and prone to human error, making automated defect detection a critical component of production pipelines.

## Key Features

- **Patch-based Classification Pipeline**: Processes 224×224 pixel patches extracted from full-resolution PCB images for focused defect localization
- **Multi-class Defect Detection**: Classifies six distinct defect categories (Open, Short, Crack, Missing Hole, Spur, Background)
- **Deep CNN Architecture**: Employs ResNet-18 backbone with transfer learning capabilities for efficient training
- **Data Augmentation**: Implements comprehensive augmentation strategies (rotation, flipping, affine transformations) to enhance model robustness
- **GPU Acceleration**: Leverages CUDA for training and inference on compatible hardware
- **Model Interpretability**: Integrates Grad-CAM visualization to highlight discriminative regions and build trust in model predictions
- **Modular Design**: Clean separation between dataset handling, model architecture, training, evaluation, and visualization components

## Dataset

### DeepPCB Dataset
This project utilizes the **DeepPCB** dataset, a benchmark for PCB defect detection containing:
- **Patch-based Format**: Pre-extracted 224×224 pixel patches from full PCB images
- **Six Defect Classes**: Open, Short, Crack, Missing Hole, Spur, and Background (non-defective)
- **Training/Validation Split**: Balanced distribution across defect categories

**Note**: The dataset is not included in this repository. To use this project, you must:
1. Request access to the DeepPCB dataset from the original source
2. Place dataset files in the `data/` directory following the expected structure
3. Update configuration paths if using alternative dataset locations

### Dataset Structure (Expected)
```
data/
├── train/
│   ├── background/
│   ├── crack/
│   ├── missing_hole/
│   ├── open/
│   ├── short/
│   └── spur/
└── test/
    ├── background/
    ├── crack/
    ├── missing_hole/
    ├── open/
    ├── short/
    └── spur/
```

## Model Architecture & Training

### ResNet-18 Classifier
- **Backbone**: ResNet-18 pretrained on ImageNet
- **Input Resolution**: 224 × 224 pixels (standard for torchvision models)
- **Output**: 6-class probability distribution (softmax)
- **Training Strategy**: Fine-tuning with frozen backbone layers and trainable classification head
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Cross-entropy loss for multi-class classification

### Training Configuration
- **Batch Size**: Configurable (typically 32–64 depending on GPU memory)
- **Epochs**: 50–100 with early stopping based on validation metrics
- **Learning Rate**: Initial 1e-4 with decay on validation plateau
- **Augmentation**: Random horizontal/vertical flips, rotations (±15°), affine transformations

## Explainability with Grad-CAM

### Why Grad-CAM?
In industrial quality control applications, model predictions must be interpretable. Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations by highlighting the image regions most influential to each classification decision. This transparency is essential for:
- **Trust & Validation**: Engineers can verify that the model focuses on relevant defect features
- **Debugging**: Identifies when models rely on spurious patterns or artifacts
- **Compliance**: Facilitates regulatory approval for automated quality assurance systems

### Implementation
- Generates class-specific attention maps for each test sample
- Overlays attention maps on original patches for visual inspection
- Supports analysis of both correct and misclassified predictions
- Enables per-class defect localization within patches

## Project Structure

```
pcb-defect-detection/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusions
├── data/                              # Dataset directory (not included)
│   ├── train/
│   └── test/
├── models/                            # Trained model checkpoints (gitignored)
│   └── resnet18_best.pth
├── outputs/                           # Visualizations and results (gitignored)
│   ├── predictions/
│   ├── grad_cam/
│   └── metrics/
├── src/                               # Source code
│   ├── __init__.py
│   ├── dataset.py                     # DeepPCBPatchDataset class
│   ├── model.py                       # Model architecture and utilities
│   ├── train.py                       # Training loop and checkpointing
│   ├── evaluate.py                    # Evaluation metrics and validation
│   ├── visualize.py                   # Grad-CAM and result visualization
│   └── config.py                      # Configuration parameters
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_pipeline.ipynb
│   └── 03_grad_cam_analysis.ipynb
└── scripts/                           # Utility scripts
    ├── download_dataset.py
    └── inference.py
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration; CPU-only is supported but slower)
- pip or conda for dependency management

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/thetushardudeja1/pcb-defect-detection.git
   cd pcb-defect-detection
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**
   - Obtain the DeepPCB dataset from the original source
   - Extract and place in the `data/` directory
   - Verify directory structure matches the expected layout

5. **Verify Installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```

## How to Run

### Training Pipeline

1. **Configure Training Parameters**
   - Edit `src/config.py` to adjust batch size, learning rate, epochs, and data paths
   - Specify device (cuda:0 or cpu) and model checkpoint location

2. **Start Training**
   - Execute the training script with your configuration
   - Monitor loss and validation accuracy in console output
   - Trained models are saved automatically to the models directory
   - Training typically requires 1–3 hours depending on dataset size and hardware

3. **Example Training Run**
   - Training uses mixed precision (if CUDA available) for memory efficiency
   - Validation is performed after each epoch
   - Best model checkpoint is preserved based on validation accuracy

### Evaluation & Metrics

1. **Run Evaluation**
   - Load trained model checkpoint from models directory
   - Compute accuracy, precision, recall, and F1-score per class
   - Generate confusion matrix for defect type analysis

2. **Interpretation**
   - Review per-class metrics to identify challenging defect types
   - Analyze class imbalance and its impact on minority classes

### Explainability & Visualization

1. **Generate Grad-CAM Visualizations**
   - Specify trained model checkpoint and test data
   - Produce attention maps for random or stratified sample selections
   - Overlay attention on original patches for interpretability analysis

2. **Output Artifacts**
   - Saves visualization images to `outputs/grad_cam/`
   - Includes heatmaps for each defect class
   - Supports batch processing for systematic analysis

3. **Analysis Workflow**
   - Visualize correct predictions to validate learned features
   - Inspect misclassifications to identify systematic errors
   - Compare attention patterns across defect types

## Technologies & Libraries

| Component | Libraries |
|-----------|-----------|
| **Deep Learning Framework** | PyTorch 2.0+ |
| **Computer Vision** | torchvision, OpenCV |
| **Numerical Computing** | NumPy |
| **Image Processing** | Pillow |
| **Visualization** | Matplotlib |
| **Model Explainability** | pytorch-grad-cam |
| **Metrics & Evaluation** | scikit-learn |

## Results & Visualization

### Model Performance
- Achieves high accuracy on standard PCB defect benchmarks
- Per-class recall demonstrates balanced performance across defect types
- Grad-CAM visualizations confirm semantic understanding of defects

### Grad-CAM Outputs
The visualization pipeline generates:
- **Class Activation Maps**: Heatmaps showing decision-relevant regions
- **Overlaid Visualizations**: Attention maps superimposed on original patches
- **Comparative Analysis**: Side-by-side comparison of correct vs. incorrect predictions

### Example Results
Sample outputs including classification predictions, attention visualizations, and performance metrics are available in the `outputs/` directory after running the evaluation pipeline.

## Usage Example

```python
from src.dataset import DeepPCBPatchDataset
from src.model import load_pretrained_resnet18
from src.visualize import visualize_grad_cam
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_pretrained_resnet18(num_classes=6).to(device)
model.load_state_dict(torch.load('models/resnet18_best.pth'))

# Load test data
dataset = DeepPCBPatchDataset(root_dir='data/test', transform=None)
test_image, label = dataset[0]

# Generate Grad-CAM visualization
visualize_grad_cam(model, test_image, device, class_names=['Open', 'Short', 'Crack', 'Missing Hole', 'Spur', 'Background'])
```

## Author

**Tushar Dudeja**  
Master's Student | Computer Vision & Deep Learning Researcher

---

*This project is intended for educational, research, and portfolio demonstration purposes. For industrial deployment, additional validation and compliance testing is required.*