# 05 - Deep Learning

Deep Learning is a branch of Machine Learning that uses **neural networks with multiple layers** to learn patterns from large amounts of data.

It is widely used in:

* Computer Vision
* Natural Language Processing
* Speech Recognition
* Image Classification
* Object Detection
* Generative AI

---

## 📚 Topics

| #  | Topic             | Notes                                    |
| -- | ----------------- | ---------------------------------------- |
| 01 | Neural Networks   | [Notes](./01-Neural-Networks/Notes.md)   |
| 02 | CNN               | [Notes](./02-CNN/Notes.md)               |
| 03 | RNN               | [Notes](./03-RNN/Notes.md)               |
| 04 | LSTM & GRU        | [Notes](./04-LSTM-GRU/Notes.md)          |
| 05 | Transfer Learning | [Notes](./05-Transfer-Learning/Notes.md) |
| 06 | Object Detection  | [Notes](./06-Object-Detection/Notes.md)  |

---

## 🧠 Learning Path

The topics are arranged from basic to more practical Deep Learning concepts.

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

---

## 01 - Neural Networks

Learn the basic building blocks of Deep Learning.

### Topics

* Neural Networks
* Neurons
* Weights
* Bias
* Activation Functions
* ReLU
* Sigmoid
* Tanh
* Softmax
* Forward Propagation
* Loss Functions
* Backpropagation
* Gradient Descent
* Learning Rate
* Epochs
* Batch Size
* Optimizers
* Overfitting
* Underfitting
* Dropout
* Train / Validation / Test

[📖 Read Notes](./01-Neural-Networks/Notes.md)

---

## 02 - CNN

**CNN = Convolutional Neural Network**

CNNs are mainly used for image and visual data.

### Topics

* CNN Architecture
* Image Representation
* Convolution
* Filters / Kernels
* Feature Maps
* Stride
* Padding
* ReLU
* Pooling
* Max Pooling
* Average Pooling
* Flattening
* Fully Connected Layers
* Image Classification
* Data Augmentation
* CNN Overfitting
* Popular CNN Architectures

### Popular Architectures

* LeNet
* AlexNet
* VGG
* GoogLeNet / Inception
* ResNet
* MobileNet
* EfficientNet

[📖 Read Notes](./02-CNN/Notes.md)

---

## 03 - RNN

**RNN = Recurrent Neural Network**

RNNs are designed to work with sequential data.

### Topics

* Sequential Data
* RNN Architecture
* Hidden State
* Recurrent Connections
* Forward Propagation
* Backpropagation Through Time
* Many-to-One
* One-to-Many
* Many-to-Many
* Sentiment Analysis
* Time Series
* Vanishing Gradient
* Exploding Gradient
* RNN Limitations

[📖 Read Notes](./03-RNN/Notes.md)

---

## 04 - LSTM & GRU

LSTM and GRU are improved versions of RNNs designed to handle long-term dependencies.

### LSTM

**LSTM = Long Short-Term Memory**

Topics:

* Cell State
* Hidden State
* Forget Gate
* Input Gate
* Output Gate
* Candidate Cell State
* LSTM Architecture
* LSTM Training

### GRU

**GRU = Gated Recurrent Unit**

Topics:

* Update Gate
* Reset Gate
* Hidden State
* GRU Architecture
* GRU Training

### Comparison

* RNN vs LSTM
* LSTM vs GRU
* Advantages and Limitations

[📖 Read Notes](./04-LSTM-GRU/Notes.md)

---

## 05 - Transfer Learning

Transfer Learning allows us to reuse knowledge learned by an existing model for a new task.

### Topics

* Transfer Learning
* Pre-trained Models
* Feature Extraction
* Fine-Tuning
* Freezing Layers
* Unfreezing Layers
* Learning Rate
* Data Augmentation
* Source Domain
* Target Domain
* Negative Transfer

### Popular Pre-trained Models

* ResNet
* VGG
* MobileNet
* EfficientNet
* DenseNet
* Inception

### Common Strategy

```text
Pre-trained Model
       ↓
Replace Classifier
       ↓
Freeze Base Model
       ↓
Train New Classifier
       ↓
Unfreeze Some Layers
       ↓
Fine-Tune
       ↓
Evaluate
```

[📖 Read Notes](./05-Transfer-Learning/Notes.md)

---

## 06 - Object Detection

Object Detection identifies objects and determines their locations in an image.

### Topics

* Object Detection
* Bounding Boxes
* Ground Truth
* Confidence Score
* IoU
* NMS
* Precision
* Recall
* F1 Score
* AP
* mAP
* mAP50
* mAP50-95
* Object Detection Dataset
* YOLO
* Real-Time Detection
* Detection Evaluation

### Popular Object Detection Models

* R-CNN
* Fast R-CNN
* Faster R-CNN
* SSD
* YOLO
* RetinaNet
* DETR

### YOLO Pipeline

```text
Image
  ↓
YOLO Model
  ↓
Bounding Boxes
  ↓
Class
  ↓
Confidence
  ↓
NMS
  ↓
Final Detection
```

[📖 Read Notes](./06-Object-Detection/Notes.md)

---

# 🛠️ Tools & Frameworks

The main tools used while learning Deep Learning are:

| Tool             | Purpose                      |
| ---------------- | ---------------------------- |
| Python           | Programming                  |
| NumPy            | Numerical Computing          |
| Pandas           | Data Handling                |
| Matplotlib       | Visualization                |
| TensorFlow       | Deep Learning                |
| Keras            | High-Level Deep Learning API |
| PyTorch          | Deep Learning Framework      |
| OpenCV           | Computer Vision              |
| Ultralytics YOLO | Object Detection             |
| Jupyter Notebook | Experimentation              |
| Google Colab     | GPU-based Training           |

---

# 💻 Frameworks

## PyTorch

PyTorch is a popular Deep Learning framework.

Used for:

* Neural Networks
* CNN
* RNN
* LSTM
* Transfer Learning
* Computer Vision

---

## TensorFlow / Keras

TensorFlow is a Deep Learning framework developed by Google.

Keras provides a high-level API for building Deep Learning models.

---

## OpenCV

OpenCV is mainly used for Computer Vision.

It can handle:

* Images
* Videos
* Webcam
* Image Processing
* Object Detection
* Real-Time Applications

---

## Ultralytics YOLO

Ultralytics provides tools for modern YOLO-based computer vision tasks.

It can be used for:

* Object Detection
* Image Classification
* Segmentation
* Pose Estimation

---

# 📊 Important Evaluation Metrics

Different Deep Learning tasks use different metrics.

### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Regression

* MAE
* MSE
* RMSE
* R²

### Object Detection

* Precision
* Recall
* F1 Score
* IoU
* AP
* mAP50
* mAP50-95

---

# 🧪 Practical Learning

While learning Deep Learning, practice should include:

```text
Theory
  ↓
Dataset Preparation
  ↓
Data Preprocessing
  ↓
Model Building
  ↓
Training
  ↓
Validation
  ↓
Evaluation
  ↓
Hyperparameter Tuning
  ↓
Deployment
```

---

# 🚀 Practical Projects

The concepts from this section can be applied to projects such as:

* Image Classification
* Face Detection
* Object Detection
* Fire Detection
* Plant Disease Detection
* Handwritten Character Recognition
* Traffic Object Detection
* Medical Image Classification

---

# 🎯 Learning Goals

After completing this section, I should be able to:

* Understand Neural Networks
* Explain how CNNs work
* Understand RNNs
* Explain LSTM and GRU
* Use Transfer Learning
* Understand Object Detection
* Explain IoU and mAP
* Train Deep Learning models
* Evaluate models
* Use PyTorch and TensorFlow/Keras
* Work with OpenCV
* Use YOLO for object detection
* Deploy Deep Learning applications

---

# 📌 Key Concepts

```text
Neural Network
    ↓
Activation Function
    ↓
Forward Propagation
    ↓
Loss Function
    ↓
Backpropagation
    ↓
Gradient Descent
    ↓
CNN / RNN
    ↓
LSTM / GRU
    ↓
Transfer Learning
    ↓
Object Detection
    ↓
Deployment
```


## 🔗 Related Sections

* [04 - Machine Learning](../04-Machine-Learning/)
* [06 - Natural Language Processing](../06-Natural-Language-Processing/)
* [07 - Computer Vision](../07-Computer-Vision/)
* [08 - AI Tools and Deployment](../08-AI-Tools-and-Deployment/)
* [09 - Generative AI and LLMs](../09-Generative-AI-and-LLMs/)
* [10 - Projects](../10-Projects/)

---


