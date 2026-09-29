# Neural Networks

## 1. What is a Neural Network?

A Neural Network is a machine learning model inspired by the way the human brain processes information.

It learns patterns from data by connecting multiple artificial neurons.

Neural Networks are widely used for:

* Classification
* Regression
* Image recognition
* Speech recognition
* Natural Language Processing
* Computer Vision

---

## 2. Basic Structure

A neural network mainly contains three types of layers:

```text
Input Layer → Hidden Layer(s) → Output Layer
```

### Input Layer

The input layer receives the features from the dataset.

Example:

```text
House Price Prediction

Area
BHK
Bathrooms
```

These features become inputs to the neural network.

### Hidden Layer

Hidden layers process the input data and learn patterns.

A neural network can have:

* One hidden layer
* Multiple hidden layers

### Output Layer

The output layer produces the final prediction.

Examples:

```text
Classification → Class probability
Regression     → Numerical value
```

---

## 3. Artificial Neuron

An artificial neuron takes inputs, applies weights and bias, and produces an output.

```text
x₁ ── w₁ ──┐
x₂ ── w₂ ──┼──> Σ + b ──> Activation ──> Output
x₃ ── w₃ ──┘
```

The basic calculation is:

```text
z = x₁w₁ + x₂w₂ + x₃w₃ + b
```

Where:

* `x` = input
* `w` = weight
* `b` = bias
* `z` = weighted sum

The weighted sum is then passed through an activation function.

---

## 4. Weights

Weights determine how important each input is.

For example:

```text
Input: Area
Weight: 0.8
```

A larger weight means that the input has a stronger influence on the calculation.

During training, the neural network updates the weights to improve predictions.

---

## 5. Bias

Bias is an additional value added to the weighted sum.

```text
z = x₁w₁ + x₂w₂ + b
```

Bias helps the model shift the activation function and learn better patterns.

---

## 6. Activation Function

An activation function determines whether and how strongly a neuron should activate.

Without activation functions, multiple neural network layers would behave like a single linear model.

Common activation functions are:

* ReLU
* Sigmoid
* Tanh
* Softmax

---

## 7. ReLU

ReLU stands for **Rectified Linear Unit**.

Formula:

```text
ReLU(x) = max(0, x)
```

Examples:

```text
x = 5   → 5
x = -3  → 0
x = 0   → 0
```

ReLU is commonly used in hidden layers.

Advantages:

* Simple
* Fast
* Helps reduce the vanishing-gradient problem compared with sigmoid/tanh in many deep networks

---

## 8. Sigmoid

The sigmoid function converts a value into a range between 0 and 1.

```text
0 < sigmoid(x) < 1
```

It is commonly used for binary classification output.

Example:

```text
0.92 → 92% probability of class 1
0.08 → 8% probability of class 1
```

Formula:

```text
σ(x) = 1 / (1 + e⁻ˣ)
```

---

## 9. Tanh

Tanh converts values into a range between:

```text
-1 and +1
```

It was commonly used in neural networks and remains useful in some architectures.

---

## 10. Softmax

Softmax converts multiple output values into probabilities whose total is 1.

It is commonly used for multi-class classification.

Example:

```text
Cat      → 0.10
Dog      → 0.75
Horse    → 0.15
```

Total:

```text
0.10 + 0.75 + 0.15 = 1.00
```

The predicted class would be:

```text
Dog
```

---

## 11. Forward Propagation

Forward propagation is the process of passing input data through the network to generate a prediction.

Basic flow:

```text
Input
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Hidden Layer
  ↓
Output Layer
  ↓
Prediction
```

Example:

```text
Input → Neural Network → Prediction
```

---

## 12. Loss Function

A loss function measures how different the prediction is from the actual value.

Example:

```text
Actual = 1
Predicted = 0.8
```

The loss function measures the error between them.

A lower loss generally means the model's predictions are closer to the target for the evaluated data.

Common loss functions include:

### Mean Squared Error (MSE)

Used mainly for regression.

```text
MSE = average((Actual - Predicted)²)
```

### Binary Cross-Entropy

Commonly used for binary classification.

### Categorical Cross-Entropy

Commonly used for multi-class classification.

---

## 13. Backpropagation

Backpropagation is an algorithm used to calculate how much each model parameter contributed to the error.

Basic process:

```text
Prediction
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Calculate Gradients
    ↓
Update Weights
```

The network uses these gradients to update its parameters.

---

## 14. Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function.

The basic idea is:

```text
Calculate loss
      ↓
Calculate gradient
      ↓
Update parameters
      ↓
Calculate loss again
      ↓
Repeat
```

A simplified update rule is:

```text
new_weight = old_weight - learning_rate × gradient
```

---

## 15. Learning Rate

The learning rate controls how large the parameter updates are.

### Very small learning rate

```text
Training → Slow
```

### Very large learning rate

```text
Training → May become unstable or overshoot
```

A suitable learning rate helps the model learn efficiently.

---

## 16. Epoch

An epoch means the model has processed the entire training dataset once.

Example:

```text
Dataset = 10,000 images

1 epoch = model processes all 10,000 images once
```

If:

```text
epochs = 10
```

the model processes the training dataset 10 times, subject to the training setup.

---

## 17. Batch Size

Batch size is the number of training samples processed before the model performs a parameter update.

Example:

```text
Dataset = 10,000 samples
Batch size = 100
```

Approximately:

```text
10,000 / 100 = 100 batches per epoch
```

---

## 18. Overfitting

Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data.

Example:

```text
Training Accuracy = 99%
Validation Accuracy = 75%
```

This can be a sign of overfitting.

Ways to reduce overfitting:

* More training data
* Data augmentation
* Dropout
* Regularization
* Early stopping
* Reduce model complexity

---

## 19. Underfitting

Underfitting occurs when the model is too simple to learn the important patterns in the data.

Example:

```text
Training Accuracy = 65%
Validation Accuracy = 63%
```

Possible solutions:

* Increase model capacity
* Train for longer
* Improve features/data
* Adjust hyperparameters

---

## 20. Dropout

Dropout is a regularization technique.

During training, randomly selected neurons are temporarily ignored.

Example:

```text
Before Dropout:

● ● ● ● ● ●

After Dropout:

● ✕ ● ● ✕ ●
```

This can help reduce overfitting.

---

## 21. Training vs Validation vs Test Data

### Training Data

Used to learn the model parameters.

### Validation Data

Used during model development to evaluate and tune the model.

### Test Data

Used for the final evaluation on unseen data.

Typical structure:

```text
Dataset
   │
   ├── Training
   ├── Validation
   └── Test
```

---

## 22. Hyperparameters

Hyperparameters are settings chosen before or during training rather than learned directly as model parameters.

Examples:

* Learning rate
* Batch size
* Number of epochs
* Number of hidden layers
* Number of neurons
* Dropout rate
* Optimizer

---

## 23. Optimizers

Optimizers determine how model parameters are updated during training.

Common optimizers:

* SGD
* Adam
* RMSprop

### Adam

Adam is widely used because it adapts the learning rate for different parameters and often provides efficient training.

---

## 24. Neural Network Example

Suppose we want to classify whether an email is spam.

Input features:

```text
Number of links
Number of suspicious words
Email length
```

The network processes these values:

```text
Input Layer
     ↓
Hidden Layer
     ↓
Hidden Layer
     ↓
Output Layer
     ↓
Spam / Not Spam
```

For binary classification, the output layer commonly contains one neuron with a sigmoid activation.

---

## 25. Neural Network Training Process

The complete training process can be summarized as:

```text
1. Give input data
        ↓
2. Forward propagation
        ↓
3. Generate prediction
        ↓
4. Calculate loss
        ↓
5. Backpropagation
        ↓
6. Calculate gradients
        ↓
7. Update weights
        ↓
8. Repeat
```

The process continues for multiple batches and epochs.

---

## 26. ANN vs Traditional Machine Learning

| Traditional ML                                | Neural Network                                                       |
| --------------------------------------------- | -------------------------------------------------------------------- |
| Often needs manual feature engineering        | Can learn useful representations                                     |
| Usually simpler                               | Can be much deeper                                                   |
| Often works well with structured/tabular data | Particularly powerful for images, audio, text and other complex data |
| Usually easier to interpret                   | Often harder to interpret                                            |
| Can work well with smaller datasets           | Deep networks often benefit from larger datasets                     |

The appropriate method depends on the dataset and problem.

---

## 27. Important Terms

| Term          | Meaning                                                  |
| ------------- | -------------------------------------------------------- |
| Neuron        | Basic computational unit                                 |
| Weight        | Learned importance of an input                           |
| Bias          | Additional learnable value                               |
| Activation    | Adds non-linearity                                       |
| Epoch         | One pass through the training dataset                    |
| Batch         | Group of samples processed together                      |
| Loss          | Measures prediction error                                |
| Gradient      | Direction/rate used for parameter updates                |
| Optimizer     | Updates model parameters                                 |
| Learning Rate | Controls update size                                     |
| Overfitting   | Performs well on training data but poorly on unseen data |
| Dropout       | Regularization technique                                 |

---

## 28. Simple Mental Model

Remember a neural network like this:

```text
INPUT
  ↓
WEIGHTS + BIAS
  ↓
ACTIVATION
  ↓
PREDICTION
  ↓
LOSS
  ↓
BACKPROPAGATION
  ↓
UPDATE WEIGHTS
  ↓
REPEAT
```

This is the basic idea behind how a neural network learns.

---

## 29. Interview Questions

### Q1. What is a neural network?

A neural network is a machine learning model made of interconnected artificial neurons that learns patterns from data.

### Q2. What is an activation function?

An activation function introduces non-linearity into a neural network and determines the output of a neuron.

### Q3. What is ReLU?

ReLU is an activation function defined as:

```text
max(0, x)
```

It is commonly used in hidden layers.

### Q4. What is backpropagation?

Backpropagation calculates gradients of the loss with respect to model parameters so that the parameters can be updated during training.

### Q5. What is an epoch?

An epoch is one complete pass through the training dataset.

### Q6. What is overfitting?

Overfitting occurs when a model performs very well on training data but does not generalize well to unseen data.

### Q7. What is the difference between a weight and a bias?

A weight controls the contribution of an input, while a bias provides an additional learnable value that shifts the neuron's activation.

---

## 30. Summary

A neural network learns by repeatedly:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradient Descent / Optimizer
  ↓
Updated Weights
```

The process repeats until the model learns useful patterns from the training data.

