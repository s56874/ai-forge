# Convolutional Neural Networks (CNN)

## 1. What is CNN?

CNN stands for **Convolutional Neural Network**.

It is a type of deep learning neural network mainly used for processing **images and visual data**.

CNNs are commonly used for:

* Image classification
* Object detection
* Face recognition
* Image segmentation
* Medical image analysis
* OCR
* Computer vision applications

---

## 2. Why CNN?

A normal neural network can process images, but it becomes inefficient when the image is large.

For example, an RGB image of:

```text
224 × 224 × 3
```

contains:

```text
150,528
```

pixel values.

A fully connected network would require many parameters.

CNNs reduce this problem by learning local patterns such as:

```text
Edges → Shapes → Textures → Objects
```

---

## 3. CNN Architecture

A basic CNN looks like:

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output
```

---

## 4. Input Image

An image is represented as numerical values.

### Grayscale image

```text
Height × Width × 1
```

Example:

```text
28 × 28 × 1
```

### RGB image

RGB has three channels:

```text
Red
Green
Blue
```

Therefore:

```text
Height × Width × 3
```

Example:

```text
224 × 224 × 3
```

---

## 5. Convolution

Convolution is the main operation in a CNN.

A small matrix called a **filter** or **kernel** moves across the image.

The filter performs calculations with the image pixels and produces a **feature map**.

Simplified:

```text
Image
  ↓
Filter / Kernel
  ↓
Feature Map
```

---

## 6. Kernel

A kernel is a small matrix containing learnable values.

Example:

```text
[ 1  0 -1 ]
[ 1  0 -1 ]
[ 1  0 -1 ]
```

During training, the CNN learns useful filter values automatically.

Different filters can learn different patterns.

For example:

```text
Filter 1 → Vertical edges
Filter 2 → Horizontal edges
Filter 3 → Corners
Filter 4 → Textures
```

---

## 7. Feature Map

The result produced after applying a filter to an image is called a **feature map**.

```text
Input Image
     ↓
  Kernel
     ↓
Feature Map
```

A feature map shows where a particular learned pattern appears in the image.

---

## 8. Stride

Stride determines how many pixels the filter moves at each step.

### Stride = 1

The filter moves one pixel at a time.

```text
→
→
→
```

### Stride = 2

The filter moves two pixels at a time.

A larger stride generally produces a smaller output feature map.

---

## 9. Padding

Padding adds extra pixels around the border of an image.

Common types:

### Valid Padding

No padding is added.

The output becomes smaller.

### Same Padding

Padding is added so that the spatial output size can be maintained when using an appropriate stride.

---

## 10. Output Size

For a 2D convolution, a common formula is:

```text
Output Size =
((N - F + 2P) / S) + 1
```

Where:

* `N` = input size
* `F` = filter size
* `P` = padding
* `S` = stride

Example:

```text
Input = 5 × 5
Filter = 3 × 3
Padding = 0
Stride = 1
```

Then:

```text
((5 - 3 + 0) / 1) + 1
= 3
```

Output:

```text
3 × 3
```

---

## 11. ReLU

After convolution, CNNs commonly use the ReLU activation function.

```text
ReLU(x) = max(0, x)
```

Example:

```text
-2 → 0
 3 → 3
-5 → 0
 7 → 7
```

ReLU introduces non-linearity into the network.

---

## 12. Pooling

Pooling reduces the spatial size of feature maps.

It helps:

* Reduce computation
* Reduce the number of parameters
* Preserve important information
* Provide some tolerance to small spatial changes

Common types:

* Max Pooling
* Average Pooling

---

## 13. Max Pooling

Max pooling selects the largest value from a region.

Example:

```text
[1  3]
[2  4]
```

Max pooling gives:

```text
4
```

A common configuration is:

```text
2 × 2 pool
Stride = 2
```

---

## 14. Average Pooling

Average pooling calculates the average value of a region.

Example:

```text
[1  3]
[2  4]
```

Average:

```text
(1 + 3 + 2 + 4) / 4
= 2.5
```

---

## 15. Flattening

After convolution and pooling, the feature maps are converted into a one-dimensional vector.

Example:

```text
Feature Maps
     ↓
Flatten
     ↓
1D Vector
```

This vector can then be passed to fully connected layers.

---

## 16. Fully Connected Layer

A fully connected layer connects neurons from the previous layer to neurons in the next layer.

It uses the extracted features to make the final prediction.

Example:

```text
CNN Features
     ↓
Fully Connected Layer
     ↓
Output
```

---

## 17. CNN for Image Classification

Suppose we want to classify:

```text
Cat
Dog
Horse
```

The CNN learns visual features progressively.

```text
Image
  ↓
Edges
  ↓
Shapes
  ↓
Textures
  ↓
Parts
  ↓
Object
  ↓
Class
```

The final layer produces class scores or probabilities.

Example:

```text
Cat    → 0.10
Dog    → 0.85
Horse  → 0.05
```

Prediction:

```text
Dog
```

---

## 18. CNN Feature Learning

One important advantage of CNNs is that the network can learn features automatically.

Early layers may learn:

```text
Edges
```

Middle layers may learn:

```text
Shapes
Textures
```

Deeper layers may learn:

```text
Object parts
```

Final layers can use these features for classification or other tasks.

---

## 19. Data Augmentation

Data augmentation creates modified versions of training images.

Examples:

* Rotation
* Flipping
* Cropping
* Scaling
* Translation
* Brightness changes

Example:

```text
Original Image
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
Flip Rotate Crop
```

Data augmentation can help improve generalization when training data is limited.

---

## 20. CNN Overfitting

CNNs can overfit, especially when the dataset is small.

Possible solutions:

* Data augmentation
* Dropout
* Regularization
* Early stopping
* More training data
* Transfer learning

---

## 21. CNN Parameters

Important CNN hyperparameters include:

* Number of filters
* Filter size
* Stride
* Padding
* Pooling size
* Learning rate
* Batch size
* Number of epochs

---

## 22. CNN vs ANN

| ANN                                           | CNN                                        |
| --------------------------------------------- | ------------------------------------------ |
| General-purpose neural network                | Designed especially for spatial data       |
| Usually requires flattened input for images   | Can directly process image tensors         |
| Many parameters for large images              | Uses local connectivity and shared weights |
| Does not explicitly exploit spatial structure | Preserves spatial relationships            |
| Less suitable for large image tasks           | Widely used for computer vision            |

---

## 23. Important CNN Architectures

Some well-known CNN architectures include:

* LeNet
* AlexNet
* VGG
* GoogLeNet / Inception
* ResNet
* MobileNet
* EfficientNet

These architectures differ in their depth, connections, computational cost, and design.

---

## 24. CNN Applications

CNNs are used in:

### Computer Vision

* Image classification
* Object detection
* Image segmentation

### Healthcare

* Medical image analysis
* X-ray analysis
* MRI analysis

### Security

* Face recognition
* Object recognition

### Agriculture

* Crop disease detection
* Plant classification

### Autonomous Systems

* Road and object detection
* Scene understanding

---

## 25. CNN and Object Detection

CNNs are also used as part of many object detection systems.

An object detector needs to answer two questions:

```text
What is the object?
Where is the object?
```

For example:

```text
Image
  ↓
Object Detection Model
  ↓
┌───────────────┐
│     Fire      │
│   0.91        │
└───────────────┘
```

The model can produce:

* Class
* Bounding box
* Confidence score

Modern object detectors such as YOLO use deep neural networks to perform object detection efficiently.

---

## 26. CNN Training Process

A CNN is trained similarly to other neural networks:

```text
Input Image
     ↓
Forward Pass
     ↓
Prediction
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Gradient Calculation
     ↓
Optimizer
     ↓
Update Weights
     ↓
Repeat
```

---

## 27. Simple Example

Suppose we have:

```text
1000 images
2 classes:

Fire
Normal
```

The CNN receives an image:

```text
Image
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
Flatten
 ↓
Dense Layer
 ↓
Output
```

The model might produce:

```text
Fire   = 0.92
Normal = 0.08
```

Prediction:

```text
Fire
```

---

## 28. Important Terms

| Term              | Meaning                                 |
| ----------------- | --------------------------------------- |
| CNN               | Convolutional Neural Network            |
| Kernel            | Small matrix used for convolution       |
| Filter            | Learnable convolution kernel            |
| Feature Map       | Output produced by a convolution        |
| Stride            | Movement of the filter                  |
| Padding           | Extra border pixels                     |
| Pooling           | Reduces spatial dimensions              |
| Flatten           | Converts feature maps to a vector       |
| ReLU              | Common activation function              |
| Epoch             | One complete pass through training data |
| Data Augmentation | Creates modified training examples      |

---

## 29. Interview Questions

### Q1. What is CNN?

CNN is a deep learning architecture commonly used for processing image and spatial data.

### Q2. Why is CNN used for images?

CNNs can learn local spatial patterns using convolution filters while sharing parameters across the image.

### Q3. What is a kernel?

A kernel is a small matrix with learnable values used to extract features from an input.

### Q4. What is pooling?

Pooling reduces the spatial dimensions of feature maps while retaining important information.

### Q5. What is Max Pooling?

Max pooling selects the maximum value from each pooling region.

### Q6. What is padding?

Padding adds values around the boundary of an input, often to control the output size.

### Q7. What is stride?

Stride determines how far the filter moves during convolution.

### Q8. What is the difference between CNN and ANN?

CNNs are designed to exploit spatial structure in data such as images, while a basic ANN does not specifically model spatial relationships.

### Q9. What is a feature map?

A feature map is the output generated by applying a convolution filter to an input.

### Q10. Can CNNs be used for object detection?

Yes. CNN-based architectures are widely used in object detection systems, including modern YOLO-based models.

---

## 30. Summary

The basic CNN pipeline is:

```text
Image
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Flatten
  ↓
Fully Connected Layer
  ↓
Prediction
```

The key idea is:

```text
CNN learns visual features automatically
from simple patterns to complex objects.
```

