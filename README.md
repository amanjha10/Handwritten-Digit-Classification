# MNIST Classification using ANN

## Overview

This project implements **handwritten digit classification** using an **Artificial Neural Network (ANN)** trained on the **MNIST dataset**. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size **28x28 pixels**.

## Dataset

The MNIST dataset is loaded using TensorFlow's `keras.datasets` module. It is split into:

- **60,000 training images**
- **10,000 test images**

## Model Architecture

This project builds a simple **Artificial Neural Network (ANN)** using Keras' Sequential API. The architecture includes:

1. **Flatten Layer**: Converts the 28x28 image into a 1D array.
2. **Dense Layer (128 neurons, ReLU activation)**: Fully connected layer.
3. **Dense Layer (10 neurons, Softmax activation)**: Outputs probabilities for each digit (0-9).

## Installation & Requirements

Ensure you have the following installed:
```sh
pip install tensorflow matplotlib
```

## Training the Model

The model is compiled with:

- **Loss Function**: Sparse Categorical Crossentropy (for multi-class classification)
- **Optimizer**: Adam (for efficient learning)
- **Evaluation Metric**: Accuracy

To train the model, run the notebook or execute:
```python
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Evaluation & Results

After training, the model is evaluated on the test set:
```python
model.evaluate(X_test, y_test)
```
The accuracy achieved is around **98%**, depending on hyperparameter tuning.

## Visualization

Sample handwritten digits from the dataset can be visualized using:
```python
import matplotlib.pyplot as plt
plt.imshow(X_train[2], cmap='gray')
plt.show()
```

## Future Enhancements

- Improve accuracy with deeper networks (CNNs instead of ANN).
- Use **data augmentation** to enhance generalization.
- Experiment with different optimizers and activation functions.

## License

This project is open-source and available for educational purposes.

