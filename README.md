# Autism Detection Deep Learning

## Overview

This repository contains a deep learning model for autism spectrum disorder (ASD) detection using VGG19 convolutional neural network combined with LSTM (Long Short-Term Memory) layers. The model is trained using K-Fold cross-validation (5-fold) for robust evaluation and improved generalization.

## Project Details

### Model Architecture

**VGG19-LSTM with K-Fold Validation**
- **Feature Extraction**: VGG19 pre-trained on ImageNet
- **Temporal Modeling**: LSTM layers (256 units, 128 units)
- **Loss Function**: Focal Loss (alpha=0.79, gamma=2.0) for handling class imbalance
- **K-Fold Configuration**: 5-fold cross-validation
- **Optimization**: Adam optimizer

### Model Performance

- **Accuracy**: Evaluated across all 5 folds
- **Cross-Validation Strategy**: 5-Fold Cross-Validation ensures robust model evaluation
- **Loss Regularization**: Dropout and L2 regularization to prevent overfitting

## Dataset

The model uses clinical imaging data for autism detection:
- **Preprocessing**: Image normalization, augmentation, and standardization
- **Train-Test Split**: 80-20 split with stratified K-Fold
- **Augmentation**: Rotation, zoom, and shift transformations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abusanny/autism-detection-deep-learning.git
   cd autism-detection-deep-learning
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

```python
python train.py --epochs 100 --batch_size 32 --folds 5
```

### Making Predictions

```python
python predict.py --image path/to/image.jpg
```

### Evaluating Model Performance

```python
python evaluate.py --model saved_model.h5
```

## Project Structure

See `PROJECT_STRUCTURE.md` for detailed directory organization.

## Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **Python 3.8+**: Programming language
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization

## Results

The model demonstrates strong performance with:
- Robust cross-validation strategy across 5 folds
- Focal loss optimization for minority class detection
- LSTM layers capturing temporal patterns in sequential data

## Future Improvements

- Integration of attention mechanisms
- Ensemble methods combining multiple architectures
- Real-time prediction API deployment
- Mobile application deployment

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is open source and available under the MIT License.

## References

- Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG19)
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
- Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection

## Contact

For questions or inquiries, please contact: abusanny@github.com
