# CNN-Based Digit Classifier

A Convolutional Neural Network (CNN) implementation for classifying handwritten digits using the MNIST dataset. This project demonstrates the implementation of a deep learning model that can accurately recognize handwritten digits (0-9) from the MNIST dataset, with additional capabilities to handle augmented and custom test data.

## Project Overview

This project implements a CNN model to classify handwritten digits from the MNIST dataset. The model uses data augmentation techniques to improve its performance and generalization capabilities. The implementation focuses on creating a robust model that can handle various transformations of handwritten digits while maintaining high accuracy.

Key aspects of the project:
- Implementation of a deep CNN architecture optimized for digit recognition
- Advanced data augmentation techniques to enhance model robustness
- Comprehensive evaluation on multiple test datasets
- Model checkpointing to save the best performing model
- Detailed performance analysis and visualization

## Features

- CNN architecture with multiple convolutional layers
  - Four Conv2D layers with increasing filter sizes (32, 32, 64, 128)
  - Strategic use of MaxPooling2D for dimensionality reduction
  - ReLU activation for non-linear feature extraction
- Data augmentation for improved model training
  - Rotation, shift, and zoom transformations
  - Helps model learn invariant features
- Batch normalization for better training stability
  - Applied after each convolutional layer
  - Helps in faster convergence and better generalization
- Dropout layers to prevent overfitting
  - 25% dropout rate after each major layer
  - Helps in reducing overfitting on training data
- Model evaluation on both MNIST and custom test datasets
  - Standard MNIST test set evaluation
  - Augmented test data evaluation
  - Custom "nearly MNIST" dataset testing

## Requirements

- Python 3.x
- TensorFlow (Latest version recommended)
- Keras (Integrated with TensorFlow)
- NumPy (For numerical computations)
- Matplotlib (For visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VenkataSaiChandrakanthReddyVemireddy/CNN-Based-Digit-Classifier.git
cd CNN-Based-Digit-Classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS
```

3. Install required packages:
```bash
pip install tensorflow keras matplotlib numpy
```

## Usage

1. Run the main script:
```bash
python coursework1.py
```

The script will:
- Load and preprocess the MNIST dataset
  - Normalize pixel values to [0,1]
  - Reshape data for CNN input
  - Convert labels to one-hot encoding
- Apply data augmentation
  - Generate additional training samples
  - Apply various transformations
- Train the CNN model
  - Use Adam optimizer with learning rate 0.001
  - Implement early stopping and learning rate reduction
  - Save checkpoints of best models
- Evaluate the model on test data
  - Calculate accuracy on standard test set
  - Test on augmented data
  - Evaluate on custom dataset
- Save the best model based on validation accuracy

## Model Architecture

The CNN model consists of:
- Input Layer (28x28x1)
- Four Conv2D blocks with increasing filters (32→32→64→128)
- Each block includes:
  - Conv2D layer with ReLU activation
  - BatchNormalization
  - MaxPooling2D (where applicable)
  - Dropout (0.25)
- Final Dense layer with softmax activation for 10-digit classification

## Data Augmentation

The model uses these augmentation techniques to improve robustness:
- Rotation: ±15 degrees
- Position shifts: ±10% in both dimensions
- Zoom: 0.7x to 1.3x

## Training Process

The training process includes:
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Validation accuracy monitoring
- Training progress visualization

## Results

The model achieves high accuracy across different test scenarios:
- Standard MNIST test set: 99.69% accuracy
- Augmented test data: 99.45% accuracy
- Custom "nearly MNIST" dataset: 98.92% accuracy

The model demonstrates strong generalization capabilities, maintaining high accuracy even with transformed and custom test data. The implementation successfully handles various digit variations while maintaining robust performance.

## License

This project is open source and available under the MIT License.

