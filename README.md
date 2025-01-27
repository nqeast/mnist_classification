# MNIST Handwritten Digit Classification
- This project demonstrates the use of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The MNIST dataset is a standard benchmark in machine learning and contains grayscale images of digits (0-9) with corresponding labels.

## Key Features
- Dataset: The MNIST dataset with 60,000 training and 10,000 testing samples.
- Model: A CNN with convolutional, max-pooling, batch normalization, and dropout layers.
- Technologies: Python, TensorFlow/Keras, NumPy, Matplotlib, and Seaborn.
- Performance: Achieved ~99% accuracy on the test dataset after 3 epochs

## Dataset
- Convolutional Layers: Extract features using 32 and 64 filters of size 3x3.
- Max-Pooling Layers: Reduce dimensionality and computational cost.
- Batch Normalization: Normalize activations for stable and faster training.
- Dense Layers: Fully connected layers with 128 neurons.
- Dropout: Regularization to prevent overfitting.
- Softmax Layer: Outputs probabilities for 10 classes.

## Results
- Accuracy: ~99.09% test accuracy.
- Most predictions were correct with very few misclassifications.
- Misclassifications were more common between visually similar digits (e.g., 5 vs. 6).
- Training and validation accuracy/loss plots show good generalization.
- Example predictions with true and predicted labels.

## Future Improvements
- Implement data augmentation to improve robustness.
- Experiment with deeper CNN architectures.
- Introduce learning rate schedules for better optimization.
- Test on more challenging datasets, such as Fashion MNIST or CIFAR-10
