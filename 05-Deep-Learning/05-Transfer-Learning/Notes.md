# Transfer Learning

## 1. What is Transfer Learning?

**Transfer Learning** is a Deep Learning technique where we use a model that has already been trained on a large dataset and adapt it to a new task.

Instead of training a neural network from scratch, we reuse the knowledge learned by an existing model.

### Simple Example

Suppose a CNN was trained on **ImageNet** with millions of images.

It has already learned features such as:

* Edges
* Lines
* Colors
* Shapes
* Textures
* Object parts

We can reuse this knowledge for a new task such as:

> Detecting different types of plant diseases.

---

# 2. Why Use Transfer Learning?

Training a deep neural network from scratch usually requires:

* Large dataset
* More computational power
* More training time
* More tuning

Transfer learning reduces these requirements.

### Without Transfer Learning

```text
Large Dataset
     ↓
Train CNN from Scratch
     ↓
Long Training
     ↓
Model
```

### With Transfer Learning

```text
Pre-trained Model
       ↓
Reuse Learned Features
       ↓
Train on New Dataset
       ↓
New Model
```

---

# 3. Pre-trained Model

A **pre-trained model** is a model that has already been trained on a large dataset.

Examples:

* ResNet
* VGG
* MobileNet
* EfficientNet
* DenseNet
* Inception
* BERT
* GPT

For computer vision, many models are pre-trained on **ImageNet**.

---

# 4. Main Idea

A CNN usually learns features in layers.

```text
Input Image
     ↓
Early Layers
     ↓
Edges
     ↓
Textures
     ↓
Shapes
     ↓
Object Parts
     ↓
High-Level Features
     ↓
Classification
```

Early layers learn general features.

Later layers learn task-specific features.

Therefore, we can often reuse the early layers and modify the final layers.

---

# 5. Transfer Learning Architecture

Suppose the original model has:

```text
Input
  ↓
Convolution Layers
  ↓
Feature Extraction
  ↓
Fully Connected Layer
  ↓
1000 Classes
```

We want only 3 classes.

We can replace the final layer:

```text
Input
  ↓
Pre-trained CNN
  ↓
Feature Extraction
  ↓
New Fully Connected Layer
  ↓
3 Classes
```

---

# 6. Two Main Approaches

There are two common approaches:

1. Feature Extraction
2. Fine-Tuning

---

# 7. Feature Extraction

In feature extraction, we use the pre-trained model as a fixed feature extractor.

The pre-trained layers are frozen.

Only the new classification layer is trained.

```text
Pre-trained Layers
       ↓
    Frozen
       ↓
Feature Extraction
       ↓
New Classifier
       ↓
   Trainable
```

### Example

Suppose ResNet was trained on ImageNet.

We freeze ResNet:

```text
ResNet Layers → Frozen
```

Then replace the final classifier:

```text
New Classifier → Trainable
```

---

# 8. What Does Freezing Mean?

**Freezing a layer** means its weights are not updated during training.

Normally:

```text
Weights
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Weights Updated
```

For a frozen layer:

```text
Weights
   ↓
Forward Pass
   ↓
Loss
   ↓
No Weight Update
```

This allows us to preserve the knowledge learned by the pre-trained model.

---

# 9. Fine-Tuning

Fine-tuning means we allow some or all pre-trained layers to continue learning using the new dataset.

Example:

```text
Early Layers → Frozen
Middle Layers → Trainable
Final Layers → Trainable
```

The model slightly adjusts its existing knowledge for the new task.

---

# 10. Feature Extraction vs Fine-Tuning

| Feature Extraction        | Fine-Tuning                   |
| ------------------------- | ----------------------------- |
| Pre-trained layers frozen | Some layers unfrozen          |
| Faster training           | More training required        |
| Needs less data           | Usually needs more data       |
| Lower computational cost  | Higher computational cost     |
| Good for small datasets   | Useful when task is different |
| Only new layers trained   | Existing weights also updated |

---

# 11. When Should We Use Feature Extraction?

Feature extraction is useful when:

* Dataset is small
* New task is similar to original task
* Limited GPU resources
* Faster training is required

Example:

```text
ImageNet → Animal Classification
```

The model has already learned many useful visual features.

---

# 12. When Should We Use Fine-Tuning?

Fine-tuning is useful when:

* Dataset is reasonably large
* New task is different from original task
* We need better adaptation
* Pre-trained features need modification

Example:

```text
ImageNet → Medical Image Classification
```

The visual patterns can be quite different, so additional adaptation may help.

---

# 13. Freezing and Unfreezing

A common strategy is:

### Step 1

Freeze the pre-trained model.

```text
Pre-trained Model → Frozen
```

### Step 2

Train the new classification layer.

```text
New Classifier → Train
```

### Step 3

Unfreeze some deeper layers.

```text
Some CNN Layers → Train
```

### Step 4

Continue training with a **small learning rate**.

This is called **fine-tuning**.

---

# 14. Why Use a Small Learning Rate?

The pre-trained model already contains useful knowledge.

If we use a very large learning rate, the model may change the existing weights too quickly.

This can destroy useful learned features.

Therefore, fine-tuning commonly uses a smaller learning rate.

Example:

```text
Initial training:
Learning Rate = 0.001

Fine-tuning:
Learning Rate = 0.0001
```

---

# 15. Transfer Learning Workflow

A common workflow is:

```text
Collect Dataset
      ↓
Prepare Images
      ↓
Train/Validation/Test Split
      ↓
Load Pre-trained Model
      ↓
Remove/Replace Classifier
      ↓
Freeze Base Model
      ↓
Train New Classifier
      ↓
Evaluate
      ↓
Unfreeze Some Layers
      ↓
Fine-Tune
      ↓
Evaluate Again
      ↓
Deploy
```

---

# 16. Data Preprocessing

The input images should generally follow the preprocessing expected by the pre-trained model.

Typical steps:

```text
Resize
   ↓
Convert to Tensor
   ↓
Normalize
   ↓
Model
```

For example:

```text
Image
 ↓
224 × 224
 ↓
Tensor
 ↓
Normalization
 ↓
ResNet
```

---

# 17. Data Augmentation

Transfer learning can also use data augmentation.

Common techniques:

* Random rotation
* Horizontal flip
* Random crop
* Zoom
* Translation
* Brightness adjustment

Example:

```text
Original Image
      ↓
 ┌────┴────┐
 ↓         ↓
Rotate    Flip
 ↓         ↓
New Training Images
```

This can help reduce overfitting when the dataset is small.

---

# 18. Popular Pre-trained CNN Models

## ResNet

**Residual Network**

Examples:

* ResNet18
* ResNet34
* ResNet50
* ResNet101
* ResNet152

Important concept:

**Residual connections / skip connections**

---

## VGG

Examples:

* VGG16
* VGG19

Simple architecture but relatively large and computationally expensive.

---

## MobileNet

Designed for lightweight applications.

Useful for:

* Mobile devices
* Edge devices
* IoT
* Low-resource systems

---

## EfficientNet

Designed to provide a good balance between:

* Accuracy
* Model size
* Computational efficiency

---

## DenseNet

Uses dense connections between layers.

Examples:

* DenseNet121
* DenseNet169
* DenseNet201

---

# 19. Transfer Learning Example

Suppose we have:

```text
Dataset:

Cat
Dog
Horse
```

A pre-trained model originally predicts:

```text
1000 ImageNet classes
```

We replace the final classifier:

```text
1000 outputs
       ↓
3 outputs
```

Now the model predicts:

```text
Cat
Dog
Horse
```

---

# 20. PyTorch Transfer Learning Example

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights="DEFAULT")

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 3)

print(model)
```

Here:

```python
requires_grad = False
```

means the original parameters are frozen.

---

# 21. Train the New Classifier

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
```

Only the new classifier is optimized.

---

# 22. Fine-Tuning in PyTorch

After initial training, we can unfreeze some layers.

Example:

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

Now `layer4` can learn from the new dataset.

Use a smaller learning rate:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)
```

---

# 23. Keras Transfer Learning Example

```python
import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False
```

Here:

```python
include_top=False
```

removes the original classification layer.

Then:

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation="softmax")
])
```

The final layer predicts 3 classes.

---

# 24. Fine-Tuning in Keras

After training the new classifier:

```python
base_model.trainable = True
```

We can freeze most layers and train only the later layers.

Example:

```python
for layer in base_model.layers[:-20]:
    layer.trainable = False
```

The last 20 layers can now be fine-tuned.

---

# 25. Global Average Pooling

Transfer learning models often use:

```text
CNN Feature Maps
       ↓
Global Average Pooling
       ↓
Dense Layer
       ↓
Output
```

Global Average Pooling converts feature maps into a smaller feature representation.

It can reduce the number of parameters compared with a large fully connected layer.

---

# 26. Transfer Learning for Small Datasets

Suppose we have only:

```text
500 images
```

Training a large CNN from scratch may be difficult.

Instead:

```text
500 Images
    ↓
Pre-trained CNN
    ↓
Reuse Learned Features
    ↓
New Classifier
```

This can make training more practical.

---

# 27. Transfer Learning for Your Fire Detection Project

Transfer learning is commonly used in object detection too.

For example:

```text
Pre-trained YOLO
      ↓
Fine-tune on Fire/Smoke Dataset
      ↓
Fire Detection Model
```

The pre-trained model already contains general visual features.

The new dataset teaches it the specific classes:

```text
Fire
Smoke
```

This is one reason pre-trained detection models are useful for projects with limited custom data.

---

# 28. Transfer Learning vs Training From Scratch

| Training From Scratch      | Transfer Learning               |
| -------------------------- | ------------------------------- |
| Starts with random weights | Starts with pre-trained weights |
| Usually requires more data | Can work with smaller datasets  |
| Longer training            | Faster training                 |
| More computation           | Less computation                |
| More difficult             | Usually easier                  |
| Learns all features again  | Reuses existing features        |

---

# 29. Advantages of Transfer Learning

### 1. Less Training Time

The model already has useful features.

### 2. Less Data Required

Useful when collecting large datasets is difficult.

### 3. Lower Computational Cost

Training can be much faster.

### 4. Good Performance

Pre-trained representations can provide a strong starting point.

### 5. Useful for Many Applications

Examples:

* Image classification
* Object detection
* NLP
* Speech recognition
* Medical imaging

---

# 30. Limitations

Transfer learning is not always the perfect solution.

### 1. Domain Difference

A model trained on normal photographs may not have ideal features for specialized scientific images.

### 2. Negative Transfer

Sometimes the knowledge from the original task can hurt performance on the new task.

### 3. Large Models

Some pre-trained models require significant memory and computation.

### 4. Preprocessing Requirements

The input format may need to match the original model's expectations.

---

# 31. What is Negative Transfer?

**Negative transfer** happens when knowledge transferred from the original task reduces performance on the new task.

Example:

```text
Source Domain:
Natural Images

Target Domain:
Very Specialized Medical Images
```

The learned features may not transfer perfectly.

---

# 32. Source Domain and Target Domain

### Source Domain

The original dataset/task used to train the pre-trained model.

Example:

```text
ImageNet
```

### Target Domain

The new dataset/task where we apply the model.

Example:

```text
Plant Disease Dataset
```

---

# 33. Important Terms

### Pre-trained Model

A model already trained on a large dataset.

### Transfer Learning

Reusing knowledge from a previous task for a new task.

### Feature Extraction

Using the pre-trained model to extract useful features.

### Fine-Tuning

Training some pre-trained layers on the new dataset.

### Freezing

Preventing model weights from being updated.

### Unfreezing

Allowing previously frozen layers to update.

### Source Domain

Original training domain.

### Target Domain

New application domain.

### Negative Transfer

Transferred knowledge negatively affects the new task.

---

# 34. Common Transfer Learning Strategy

A practical strategy is:

```text
Step 1
Load Pre-trained Model

        ↓

Step 2
Replace Final Layer

        ↓

Step 3
Freeze Base Model

        ↓

Step 4
Train New Classifier

        ↓

Step 5
Evaluate

        ↓

Step 6
Unfreeze Some Later Layers

        ↓

Step 7
Use Small Learning Rate

        ↓

Step 8
Fine-Tune

        ↓

Step 9
Evaluate Again
```

---

# 35. Interview Questions

### Q1. What is transfer learning?

Transfer learning is a technique where knowledge from a pre-trained model is reused for a new task.

---

### Q2. Why is transfer learning useful?

It reduces training time, data requirements, and computational requirements.

---

### Q3. What is a pre-trained model?

A model that has already been trained on a large dataset.

---

### Q4. What is feature extraction?

Using a pre-trained model to extract useful features while keeping its learned weights fixed.

---

### Q5. What is fine-tuning?

Fine-tuning means training some of the pre-trained layers on the new dataset.

---

### Q6. What does freezing a layer mean?

It means the layer's weights are not updated during training.

---

### Q7. Why use a small learning rate during fine-tuning?

To make small adjustments to the pre-trained weights without changing useful learned features too aggressively.

---

### Q8. What is negative transfer?

Negative transfer occurs when transferred knowledge hurts performance on the target task.

---

### Q9. Name some popular pre-trained models.

Examples:

* ResNet
* VGG
* MobileNet
* EfficientNet
* DenseNet
* Inception

---

### Q10. What is the difference between feature extraction and fine-tuning?

Feature extraction keeps the pre-trained layers frozen, while fine-tuning allows some pre-trained layers to learn from the new dataset.

---

# 36. Simple Real-World Example

Suppose we want to classify:

```text
Healthy Leaf
Diseased Leaf
```

We have only:

```text
1,000 images
```

Instead of creating a CNN from scratch:

```text
1,000 images
      ↓
CNN from scratch
      ↓
Training
```

we can use:

```text
Pre-trained ResNet
       ↓
Freeze Layers
       ↓
Replace Classifier
       ↓
Train
       ↓
Fine-Tune
       ↓
Leaf Disease Model
```

---

# 37. Key Takeaway

The main idea of transfer learning is:

> **Do not learn everything from zero when a model has already learned useful knowledge.**

The common process is:

```text
Pre-trained Model
       ↓
Reuse Knowledge
       ↓
Replace Classifier
       ↓
Freeze
       ↓
Train
       ↓
Fine-Tune
       ↓
Evaluate
       ↓
Deploy
```

---

# 38. Summary

* Transfer learning reuses knowledge from an existing model.
* Pre-trained models are trained on large datasets.
* CNNs are commonly transferred for computer vision tasks.
* Feature extraction keeps the base model frozen.
* Fine-tuning allows some pre-trained layers to learn.
* Freezing prevents weight updates.
* Unfreezing allows weight updates.
* Fine-tuning usually uses a small learning rate.
* Data augmentation can help with small datasets.
* ResNet, VGG, MobileNet, EfficientNet, and DenseNet are popular pre-trained CNNs.
* Transfer learning can reduce training time and data requirements.
* Negative transfer can occur when the source and target tasks are very different.
* Transfer learning is widely used in image classification and object detection.

---

## Learning Flow

```text
Neural Networks
       ↓
      CNN
       ↓
      RNN
       ↓
   LSTM / GRU
       ↓
Transfer Learning
       ↓
Object Detection
```



