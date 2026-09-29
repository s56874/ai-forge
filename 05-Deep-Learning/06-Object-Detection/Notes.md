# Object Detection

## 1. What is Object Detection?

**Object Detection** is a Computer Vision task used to:

1. **Find objects** in an image.
2. **Identify what the objects are.**

Unlike image classification, object detection tells us **where** the object is.

### Example

Suppose an image contains:

```text
Car
Person
Dog
```

An object detection model can produce:

```text
Person → Bounding Box → Confidence: 0.94
Car    → Bounding Box → Confidence: 0.91
Dog    → Bounding Box → Confidence: 0.87
```

---

# 2. Classification vs Object Detection

### Image Classification

Answers:

> What is in this image?

```text
Image
  ↓
Model
  ↓
Dog
```

### Object Detection

Answers:

> What objects are present and where are they?

```text
Image
  ↓
Model
  ↓
Dog → Box
Car → Box
Person → Box
```

---

# 3. Object Detection vs Image Classification

| Classification          | Object Detection            |
| ----------------------- | --------------------------- |
| Identifies image/class  | Identifies objects          |
| Usually one prediction  | Can detect multiple objects |
| No location information | Gives object location       |
| No bounding box         | Uses bounding boxes         |

---

# 4. What is a Bounding Box?

A **bounding box** is a rectangle around an object.

Example:

```text
+-------------------------+
|                         |
|      +----------+       |
|      |   DOG    |       |
|      |          |       |
|      +----------+       |
|                         |
+-------------------------+
```

The box tells the model where the object is located.

---

# 5. Bounding Box Coordinates

A bounding box can be represented using:

```text
x1
y1
x2
y2
```

Where:

* `x1` = left coordinate
* `y1` = top coordinate
* `x2` = right coordinate
* `y2` = bottom coordinate

Example:

```text
(x1, y1) = (100, 50)
(x2, y2) = (300, 250)
```

---

# 6. Center-Based Bounding Box

Another common representation is:

```text
x_center
y_center
width
height
```

For YOLO, normalized coordinates are commonly used.

Example:

```text
0.50 0.40 0.30 0.20
```

The values are generally between:

```text
0 and 1
```

---

# 7. Object Detection Output

A detector usually produces:

```text
Class
Bounding Box
Confidence Score
```

Example:

```text
Fire
Box: (120, 80, 350, 300)
Confidence: 0.91
```

---

# 8. Confidence Score

The **confidence score** indicates how confident the model is about a prediction.

Example:

```text
Fire → 0.92
Smoke → 0.81
Person → 0.34
```

If we use a threshold:

```text
Confidence Threshold = 0.70
```

Then:

```text
0.92 → Keep
0.81 → Keep
0.34 → Remove
```

---

# 9. Object Detection Pipeline

A simple pipeline is:

```text
Input Image
     ↓
Preprocessing
     ↓
Object Detection Model
     ↓
Bounding Boxes
     ↓
Class Predictions
     ↓
Confidence Scores
     ↓
NMS
     ↓
Final Predictions
```

---

# 10. What is IoU?

**IoU = Intersection over Union**

IoU measures how much two bounding boxes overlap.

It is commonly used to compare:

* Ground-truth box
* Predicted box

Formula:

```text
IoU = Area of Intersection
      ---------------------
       Area of Union
```

---

# 11. IoU Example

Suppose:

```text
Ground Truth Box
       +
       |
       |   +---------+
       |   |         |
       +---|---------|
           |         |
           +---------+

Predicted Box
```

If the boxes overlap strongly:

```text
IoU → High
```

If they barely overlap:

```text
IoU → Low
```

IoU ranges from:

```text
0 → No overlap
1 → Perfect overlap
```

---

# 12. Why is IoU Important?

IoU helps determine whether a prediction correctly matches the ground-truth object.

For example:

```text
IoU = 0.80
```

This means the predicted box overlaps the ground-truth box significantly.

A particular evaluation may define a threshold such as:

```text
IoU ≥ 0.50
```

for considering a detection a match.

---

# 13. Ground Truth

**Ground truth** means the correct answer provided in the dataset.

Example:

```text
Image
 ↓
Ground Truth:
Fire
Box = (100, 80, 300, 250)
```

The model tries to predict the same object and location.

---

# 14. Object Detection Dataset

A typical dataset contains:

```text
dataset/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── image3.txt
```

Each image has a corresponding label file.

---

# 15. YOLO Label Format

YOLO commonly uses:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.50 0.40 0.30 0.20
```

Meaning:

```text
Class ID = 0
Center X = 0.50
Center Y = 0.40
Width    = 0.30
Height   = 0.20
```

Coordinates are normalized relative to the image dimensions.

---

# 16. Multiple Objects

An image can contain multiple objects.

Example:

```text
0 0.30 0.40 0.20 0.30
1 0.70 0.50 0.25 0.40
```

This means the image contains two labeled objects.

---

# 17. Popular Object Detection Models

Some well-known object detection architectures include:

* R-CNN
* Fast R-CNN
* Faster R-CNN
* SSD
* YOLO
* RetinaNet
* DETR

---

# 18. YOLO

**YOLO = You Only Look Once**

YOLO is a popular family of object detection models designed for fast object detection.

Basic idea:

```text
Image
  ↓
YOLO
  ↓
Objects + Boxes + Confidence
```

YOLO is widely used for real-time applications.

---

# 19. Why YOLO is Popular

YOLO is commonly used because it can provide:

* Fast inference
* Object localization
* Classification
* Real-time detection
* Easy deployment
* GPU acceleration

Applications include:

* Fire detection
* Vehicle detection
* Person detection
* Traffic monitoring
* Industrial inspection
* Surveillance
* Robotics

---

# 20. YOLO Workflow

```text
Image
  ↓
YOLO Model
  ↓
Feature Extraction
  ↓
Object Predictions
  ↓
Bounding Boxes
  ↓
Class + Confidence
  ↓
NMS
  ↓
Final Detection
```

---

# 21. NMS

**NMS = Non-Maximum Suppression**

Sometimes the model predicts multiple overlapping boxes for the same object.

Example:

```text
      +-----------+
      |           |
   +--|-----------|--+
   |  |   FIRE    |  |
   |  +-----------+  |
   +-----------------+
```

We do not want many boxes for the same object.

NMS keeps the best prediction and removes highly overlapping lower-confidence predictions.

---

# 22. NMS Process

Suppose predictions are:

```text
Box A → 0.95
Box B → 0.88
Box C → 0.62
```

If they refer to the same object, NMS may:

```text
Keep Box A
Remove Box B
Remove Box C
```

based on the chosen overlap criteria.

---

# 23. Precision

Precision answers:

> Out of all predicted positive detections, how many were actually correct?

Formula:

```text
Precision = TP
            --------
            TP + FP
```

Where:

* TP = True Positive
* FP = False Positive

---

# 24. Recall

Recall answers:

> Out of all actual objects, how many did the model detect?

Formula:

```text
Recall = TP
         --------
         TP + FN
```

Where:

* TP = True Positive
* FN = False Negative

---

# 25. Example of Precision and Recall

Suppose the model detects:

```text
100 objects
```

But:

```text
80 = Correct
20 = Incorrect
```

Then:

```text
Precision = 80 / 100
          = 0.80
          = 80%
```

Suppose there were actually:

```text
100 objects
```

and the model detected:

```text
80
```

Then:

```text
Recall = 80 / 100
       = 80%
```

---

# 26. F1 Score

F1 Score combines precision and recall.

Formula:

```text
F1 = 2 × Precision × Recall
     ------------------------
     Precision + Recall
```

F1 is useful when we want a balance between precision and recall.

---

# 27. mAP

**mAP = mean Average Precision**

It is one of the important evaluation metrics for object detection.

mAP evaluates detection performance across classes and detection thresholds.

For example:

```text
mAP50
mAP50-95
```

are commonly reported for modern object detection models.

---

# 28. mAP50

**mAP50** means mean Average Precision at an IoU threshold of 0.50.

A higher value generally indicates better detection performance under that evaluation setting.

Example:

```text
mAP50 = 0.75
```

means:

```text
75%
```

when expressed as a percentage.

---

# 29. mAP50-95

This metric averages AP across multiple IoU thresholds:

```text
0.50
0.55
0.60
...
0.95
```

It is stricter than using only IoU 0.50.

Example:

```text
mAP50     = 0.80
mAP50-95  = 0.52
```

The second metric evaluates localization more strictly.

---

# 30. Confusion Matrix

A confusion matrix helps analyze classification errors.

For a binary detection task:

```text
                Predicted
              Fire   Normal

Actual Fire    TP      FN

Actual Normal  FP      TN
```

Where:

* TP = True Positive
* TN = True Negative
* FP = False Positive
* FN = False Negative

---

# 31. Fire Detection Example

For your FireWatch project:

```text
Classes:

0 → Smoke
1 → Fire
```

The model receives:

```text
Camera Frame
      ↓
YOLO
      ↓
Fire / Smoke
      ↓
Bounding Box
      ↓
Confidence
```

If:

```text
Fire Confidence = 0.82
```

and threshold is:

```text
0.70
```

the system can trigger an alert.

---

# 32. Object Detection Training

Typical training process:

```text
Dataset
   ↓
Label Images
   ↓
Train/Validation Split
   ↓
Load Pre-trained Model
   ↓
Train
   ↓
Validate
   ↓
Evaluate
   ↓
Tune
   ↓
Test
```

---

# 33. Train / Validation / Test

A common split might be:

```text
Training     → 70–80%
Validation   → 10–20%
Testing      → 10–20%
```

The exact split depends on the dataset.

### Training Set

Used to learn model parameters.

### Validation Set

Used during development and tuning.

### Test Set

Used for final evaluation.

---

# 34. Data Augmentation

Object detection models can benefit from augmentation.

Examples:

* Horizontal flipping
* Scaling
* Cropping
* Rotation
* Translation
* Brightness changes
* Contrast changes

But augmentation should remain realistic.

For example, excessive transformations can create unrealistic training images.

---

# 35. Overfitting in Object Detection

Overfitting happens when the model performs well on training images but poorly on unseen images.

Example:

```text
Training mAP → 0.95
Validation mAP → 0.60
```

Possible causes:

* Small dataset
* Duplicate images
* Poor dataset diversity
* Too many training epochs
* Incorrect labels

---

# 36. How to Improve Object Detection

Possible methods:

### 1. Improve Dataset

Add diverse images.

### 2. Improve Labels

Check bounding boxes carefully.

### 3. Remove Bad Images

Remove:

* Corrupted images
* Duplicate images
* Unrealistic images
* Incorrect labels

### 4. Data Augmentation

Increase realistic variation.

### 5. Transfer Learning

Start from a pre-trained model.

### 6. Tune Hyperparameters

Examples:

* Learning rate
* Batch size
* Image size
* Number of epochs

---

# 37. Dataset Quality Is Important

A large dataset is not automatically a good dataset.

For example:

```text
10,000 Poor Images
```

may perform worse than:

```text
3,000 High-Quality Diverse Images
```

Important factors include:

* Correct labels
* Diverse conditions
* Different object sizes
* Different backgrounds
* Different lighting
* Different camera angles

---

# 38. Small Object Detection

Detecting small objects can be difficult.

Example:

```text
Large Fire
→ Easier

Very Small Fire
→ More Difficult
```

Possible approaches include:

* Higher image resolution
* Better dataset
* More small-object examples
* Appropriate model architecture
* Careful augmentation

---

# 39. Real-Time Object Detection

Real-time detection means processing frames quickly enough for an interactive application.

Example:

```text
Camera
  ↓
Frame
  ↓
YOLO
  ↓
Prediction
  ↓
Display
  ↓
Next Frame
```

Performance can depend on:

* GPU
* CPU
* Image resolution
* Model size
* Batch size
* Number of objects

---

# 40. Object Detection with OpenCV

OpenCV can be used to:

* Read images
* Read videos
* Access cameras
* Draw bounding boxes
* Display predictions
* Save frames

Example:

```python
import cv2

image = cv2.imread("image.jpg")

cv2.rectangle(
    image,
    (100, 100),
    (300, 300),
    (0, 255, 0),
    2
)

cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# 41. YOLO with Python

Using Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("image.jpg")

for result in results:
    result.show()
```

Here:

```text
best.pt
```

contains the trained model weights.

---

# 42. YOLO Prediction on Webcam

A simple example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model(source=0, show=True)
```

Here:

```text
source=0
```

usually refers to the default webcam.

---

# 43. Detection Confidence Threshold

Example:

```python
results = model(
    "image.jpg",
    conf=0.70
)
```

This tells the detector to consider predictions above the selected confidence threshold.

The best threshold depends on the application and validation results.

---

# 44. Object Detection Deployment

A practical application can be:

```text
Camera
   ↓
OpenCV
   ↓
YOLO
   ↓
Detection
   ↓
Business Logic
   ↓
Alert / Database / Dashboard
```

For example:

```text
Camera
   ↓
Fire detected
   ↓
Confidence > threshold
   ↓
Save frame
   ↓
Record event
   ↓
Trigger alert
```

---

# 45. Common Object Detection Applications

### Security

Person and vehicle detection.

### Agriculture

Crop and disease detection.

### Healthcare

Medical object detection.

### Manufacturing

Defect detection.

### Traffic

Vehicle and pedestrian detection.

### Fire Safety

Fire and smoke detection.

### Robotics

Object recognition and localization.

---

# 46. Object Detection vs Segmentation

### Object Detection

Finds objects using bounding boxes.

```text
+---------+
|  Object |
+---------+
```

### Image Segmentation

Identifies pixels belonging to an object.

```text
Pixel-level object region
```

Segmentation provides more detailed localization.

---

# 47. Object Detection vs Instance Segmentation

### Object Detection

```text
Person → Bounding Box
```

### Instance Segmentation

```text
Person → Exact Pixel Mask
```

If there are two people, instance segmentation can create a separate mask for each person.

---

# 48. Detection Pipeline Summary

```text
Image
  ↓
Preprocessing
  ↓
Feature Extraction
  ↓
Object Predictions
  ↓
Bounding Boxes
  ↓
Class Scores
  ↓
Confidence Filtering
  ↓
NMS
  ↓
Final Detections
```

---

# 49. Important Terms

### Bounding Box

Rectangle around an object.

### Ground Truth

Correct object annotation.

### Confidence

Model's confidence in a prediction.

### IoU

Measures overlap between two boxes.

### NMS

Removes duplicate overlapping detections.

### Precision

Measures correctness of positive predictions.

### Recall

Measures how many actual objects were detected.

### F1 Score

Balance between precision and recall.

### AP

Average Precision for a class.

### mAP

Mean Average Precision across classes/evaluation settings.

### YOLO

A popular object detection model family.

---

# 50. Interview Questions

### Q1. What is object detection?

Object detection identifies objects in an image and determines their locations using bounding boxes.

### Q2. What is a bounding box?

A bounding box is a rectangular region used to represent the location of an object.

### Q3. What is IoU?

IoU measures the overlap between predicted and ground-truth bounding boxes.

### Q4. What is NMS?

NMS removes duplicate overlapping predictions and keeps the more relevant detection.

### Q5. What is YOLO?

YOLO stands for You Only Look Once and is a popular object detection model family designed for fast detection.

### Q6. What is mAP?

mAP stands for mean Average Precision and is a common object detection evaluation metric.

### Q7. What is the difference between precision and recall?

Precision measures how many predicted detections are correct, while recall measures how many actual objects were detected.

### Q8. What is mAP50?

mAP50 evaluates mean Average Precision using an IoU threshold of 0.50.

### Q9. What is mAP50-95?

It averages AP over IoU thresholds from 0.50 to 0.95 in increments of 0.05.

### Q10. What is the difference between object detection and classification?

Classification predicts what an image contains, while object detection predicts both what objects are present and where they are located.

### Q11. Why is transfer learning useful in object detection?

A pre-trained model already contains useful visual features, so it can be adapted to a custom detection dataset with less training than starting from scratch.

### Q12. What causes false positives?

Possible causes include:

* Poor training data
* Incorrect labels
* Similar-looking objects
* Low confidence threshold
* Background confusion

### Q13. What causes false negatives?

Possible causes include:

* Poor image quality
* Small objects
* Occlusion
* Insufficient training examples
* High confidence threshold

---

# 51. Key Takeaway

Object detection answers two questions:

```text
WHAT is the object?
        +
WHERE is the object?
```

The basic process is:

```text
Image
 ↓
Model
 ↓
Bounding Box
 ↓
Class
 ↓
Confidence
 ↓
NMS
 ↓
Final Detection
```

For practical AI projects, **YOLO + OpenCV + Python** is a common combination for building real-time detection systems.

---

# 52. Summary

* Object detection identifies and locates objects.
* Bounding boxes represent object locations.
* Ground truth contains the correct annotations.
* Confidence represents prediction confidence.
* IoU measures bounding-box overlap.
* NMS removes duplicate predictions.
* Precision measures prediction correctness.
* Recall measures detection coverage.
* F1 combines precision and recall.
* AP measures performance for a class.
* mAP summarizes Average Precision.
* YOLO is widely used for fast object detection.
* Transfer learning can reduce training requirements.
* Dataset quality has a major impact on performance.
* OpenCV can handle images, videos, and cameras.
* Object detection is widely used in real-world AI applications.

---



**Next section:** `06-Natural-Language-Processing/`

