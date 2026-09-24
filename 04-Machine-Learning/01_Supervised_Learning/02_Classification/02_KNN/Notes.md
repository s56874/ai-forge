# KNN (K-Nearest Neighbors) — Complete Notes

## 1. What is KNN?

**KNN = K-Nearest Neighbors**

KNN is a **supervised machine learning algorithm** used mainly for:

* Classification
* Regression

For classification, KNN predicts the class of a new data point by looking at its **K nearest training data points**.

### Example

For Loan Approval:

```text
0 = Loan Not Approved
1 = Loan Approved
```

If most of the nearest customers have `loan_approved = 1`, KNN predicts:

```text
Loan Approved
```

### Important Point

KNN is called a **distance-based algorithm** because it uses distance between data points.

---

# 2. Basic Idea

The basic idea of KNN is:

> "A new data point is likely to belong to the same class as its nearby data points."

Suppose we have a new customer:

```text
Age = 30
Salary = 60000
Credit Score = 750
Loan Amount = 300000
```

KNN finds the nearest customers from the training dataset.

Suppose:

```text
K = 5
```

The 5 nearest customers are:

```text
Customer 1 → Approved
Customer 2 → Approved
Customer 3 → Not Approved
Customer 4 → Approved
Customer 5 → Approved
```

Voting:

```text
Approved      = 4
Not Approved  = 1
```

Therefore:

```text
Prediction = Loan Approved
```

---

# 3. Mathematical Formula

KNN needs a method to calculate the distance between two data points.

The most common distance is **Euclidean Distance**.

## Euclidean Distance

For two points:

```text
A = (a1, a2, ..., an)
B = (b1, b2, ..., bn)
```

Formula:

```text
d(A,B) = √[(a1-b1)² + (a2-b2)² + ... + (an-bn)²]
```

For two features:

```text
A = (x1, y1)
B = (x2, y2)
```

```text
Distance = √[(x1-x2)² + (y1-y2)²]
```

### Example

```text
A = (2, 3)
B = (5, 7)
```

```text
Distance = √[(2-5)² + (3-7)²]

         = √[9 + 16]

         = √25

         = 5
```

Smaller distance means:

```text
More similar
```

---

# 4. What is K?

`K` represents the number of nearest neighbors used for prediction.

Example:

```python
K = 5
```

means:

> KNN looks at the 5 nearest training points.

### Small K

Example:

```text
K = 1
```

The prediction depends only on the closest point.

Advantages:

* Can capture local patterns

Disadvantage:

* Sensitive to noise
* Can overfit

### Large K

Example:

```text
K = 20
```

The algorithm considers many neighbors.

Advantages:

* More stable
* Less sensitive to individual noisy points

Disadvantages:

* Can become too generalized
* Can underfit

### Important Interview Point

```text
Small K → More complex → Higher overfitting risk

Large K → Smoother → Higher underfitting risk
```

---

# 5. What are Nearest Neighbors?

Nearest neighbors are the training data points that have the **smallest distance** from the new data point.

Example:

```text
New customer

Distance to Customer A = 2.1
Distance to Customer B = 4.8
Distance to Customer C = 1.5
Distance to Customer D = 3.0
```

Sorted:

```text
Customer C → 1.5
Customer A → 2.1
Customer D → 3.0
Customer B → 4.8
```

If:

```text
K = 2
```

KNN selects:

```text
Customer C
Customer A
```

These are the two nearest neighbors.

---

# 6. Important KNN Parameters

The main parameters of `KNeighborsClassifier` are:

## 1. `n_neighbors`

Controls the value of K.

```python
n_neighbors=5
```

Means:

```text
Use 5 nearest neighbors.
```

---

## 2. `weights`

Controls how neighbors contribute to the prediction.

### `weights="uniform"`

Every neighbor gets equal importance.

```text
Neighbor 1 → equal vote
Neighbor 2 → equal vote
Neighbor 3 → equal vote
```

### `weights="distance"`

Closer neighbors receive more importance.

```text
Very close neighbor → higher influence
Far neighbor         → lower influence
```

---

## 3. `metric`

Defines how distance is calculated.

Common options:

```text
euclidean
manhattan
minkowski
```

---

## 4. `p`

Used with the Minkowski distance.

```text
p = 2 → Euclidean distance
p = 1 → Manhattan distance
```

---

## 5. `algorithm`

Controls how nearest neighbors are searched.

Common options:

```text
auto
ball_tree
kd_tree
brute
```

For a small dataset, `auto` is usually sufficient.

---

# 7. Why Feature Scaling is Important

Feature scaling is **very important for KNN**.

Why?

Because KNN uses distance.

Our Loan Approval dataset contains:

```text
Age
Salary
Years Experience
Credit Score
Loan Amount
```

Their ranges can be very different.

For example:

```text
Age           → 20–60
Salary        → 20,000–150,000
Credit Score  → 300–850
Loan Amount   → 100,000–1,000,000
```

Without scaling, large numerical values can dominate the distance calculation.

Therefore, we use:

```python
StandardScaler()
```

Standardization approximately converts features to:

```text
Mean = 0
Standard deviation = 1
```

### Important Interview Answer

> "Feature scaling is important in KNN because KNN calculates distances between data points. Without scaling, features with larger numerical ranges can dominate the distance."

---

# 8. Simple KNN Example

The basic workflow is:

```text
Training Data
      ↓
Scale Features
      ↓
Choose K
      ↓
Calculate Distances
      ↓
Find K Nearest Neighbors
      ↓
Majority Voting
      ↓
Prediction
```

For example:

```text
K = 5
```

If the neighbors vote:

```text
Approved      → 4
Not Approved  → 1
```

Prediction:

```text
Approved
```

---

# 9. Loan Approval Dataset

Our dataset contains:

```text
30 rows
6 columns
```

Columns:

```text
age
salary
years_experience
credit_score
loan_amount
loan_approved
```

Target:

```text
loan_approved
```

Meaning:

```text
0 → Loan Not Approved
1 → Loan Approved
```

Features:

```text
age
salary
years_experience
credit_score
loan_amount
```

### X and y

```text
X = Input Features

y = Target
```

Therefore:

```python
X = df[
    [
        "age",
        "salary",
        "years_experience",
        "credit_score",
        "loan_amount"
    ]
]

y = df["loan_approved"]
```

There are:

* No missing values
* Numerical features
* No categorical encoding required

---

# 10. How KNN Makes a Prediction

Suppose we have a new customer.

KNN performs these steps:

### Step 1 — Take the new customer

```text
New Customer
```

### Step 2 — Calculate distance

Calculate distance from the new customer to training customers.

### Step 3 — Sort distances

Nearest points come first.

### Step 4 — Select K neighbors

For:

```text
K = 5
```

select the 5 closest customers.

### Step 5 — Voting

Example:

```text
Approved
Approved
Not Approved
Approved
Approved
```

Count:

```text
Approved      = 4
Not Approved  = 1
```

### Step 6 — Final Prediction

```text
Loan Approved
```

### Main idea

```text
Distance → Neighbors → Voting → Prediction
```

---

# 11. `predict()` vs `predict_proba()`

## `predict()`

Returns the final predicted class.

Example:

```text
0 → Not Approved
1 → Approved
```

Conceptually:

```python
y_pred = model.predict(X_test)
```

---

## `predict_proba()`

Returns the estimated probability for each class based on the neighbors.

For example:

```text
[0.20, 0.80]
```

means approximately:

```text
Class 0 → 20%
Class 1 → 80%
```

The exact probabilities depend on the KNN configuration, especially the neighbor weighting.

### Important Difference

```text
predict()
→ Final class

predict_proba()
→ Class probabilities
```

Unlike Logistic Regression, **KNN's default classification mechanism is majority voting among neighbors**, not a sigmoid followed by a 0.5 threshold.

---

# 12. Best Graph

For KNN, a very useful graph is:

## Accuracy vs K

We can test different values:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
...
```

Then compare their validation accuracy.

Conceptually:

```text
Accuracy
   |
   |       *
   |   *       *
   | *           *
   |________________
       K values
```

This helps us understand how changing K affects performance.

### Another important graph

Confusion Matrix:

```text
Actual vs Predicted
```

It helps us understand:

```text
Correct predictions
False predictions
```

---

# 13. Confusion Matrix

For binary Loan Approval:

```text
                 Predicted
              0          1

Actual 0     TN         FP

Actual 1     FN         TP
```

Where:

### TN — True Negative

Actual:

```text
0
```

Predicted:

```text
0
```

Correctly rejected loan.

### TP — True Positive

Actual:

```text
1
```

Predicted:

```text
1
```

Correctly approved loan.

### FP — False Positive

Actual:

```text
0
```

Predicted:

```text
1
```

Model approved a loan that should be rejected according to the label.

### FN — False Negative

Actual:

```text
1
```

Predicted:

```text
0
```

Model rejected a loan that should be approved according to the label.

---

# 14. Classification Evaluation Metrics

## 1. Accuracy

Measures overall correct predictions.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

## 2. Precision

Out of all predicted positive cases, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

---

## 3. Recall

Out of all actual positive cases, how many did the model identify?

```text
Recall = TP / (TP + FN)
```

---

## 4. F1 Score

Combination of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

---

## 5. ROC-AUC

Measures how well the model separates the two classes across probability thresholds.

For KNN, we can use:

```python
model.predict_proba(X_test)[:, 1]
```

### Important

Because our dataset has only **30 rows**, evaluation results can change significantly depending on the train-test split.

---

# 15. Choosing the Best K

We should not randomly choose K.

We can test multiple values:

```text
K = 3
K = 5
K = 7
K = 9
```

and use **cross-validation** to compare them.

For example:

```text
K = 3 → CV Accuracy
K = 5 → CV Accuracy
K = 7 → CV Accuracy
K = 9 → CV Accuracy
```

We select the configuration based on the validation results rather than choosing K only because it gives a good result on the test set.

### Important

The test set should remain separate and should be used for final evaluation.

---

# 16. Complete KNN Model

Our complete workflow is:

```text
Loan Dataset
      ↓
Select Features
      ↓
Select Target
      ↓
Train-Test Split
      ↓
StandardScaler
      ↓
KNN Classifier
      ↓
Try Different K Values
      ↓
Cross-Validation
      ↓
Select Best Parameters
      ↓
Train Best Model
      ↓
Predict
      ↓
Evaluate
```

For reliable tuning, we use a `Pipeline`:

```text
Scaler → KNN
```

This is especially important during cross-validation because the scaler should be fitted only using the training portion of each fold.

---

# 17. Complete KNN Code

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("loan_approval.csv")

# Features
X = df[
    [
        "age",
        "salary",
        "years_experience",
        "credit_score",
        "loan_amount"
    ]
]

# Target
y = df["loan_approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Parameters to test
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Predictions
y_pred = model.predict(X_test)

# Probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV Accuracy:")
print(grid_search.best_score_)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred))

print("\nRecall:")
print(recall_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Graph
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("KNN Loan Approval - Confusion Matrix")
plt.show()
```

### Code Flow

```text
CSV
 ↓
X and y
 ↓
Train-Test Split
 ↓
StandardScaler
 ↓
KNN
 ↓
GridSearchCV
 ↓
Best K + Parameters
 ↓
Prediction
 ↓
Evaluation
```

---

# 18. KNN Overfitting and Underfitting

## Overfitting

Usually more likely with a very small K.

Example:

```text
K = 1
```

The model can become highly sensitive to individual training points and noise.

```text
Training Performance → Very High
Validation Performance → Lower
```

---

## Underfitting

Can happen with a very large K.

The model becomes too smooth and may ignore local patterns.

```text
Training Performance → Lower
Validation Performance → Lower
```

### Simple Rule

```text
Very Small K → Overfitting Risk

Very Large K → Underfitting Risk
```

The best K should be selected using validation/cross-validation rather than this rule alone.

---

# 19. KNN Distance Metrics and Weights

## Euclidean Distance

Straight-line distance.

```text
√[(x1-x2)² + (y1-y2)²]
```

Common default choice.

---

## Manhattan Distance

Distance along dimensions:

```text
|x1-x2| + |y1-y2|
```

---

## Minkowski Distance

Generalized distance:

```text
[Σ |xi - yi|^p]^(1/p)
```

Special cases:

```text
p = 1 → Manhattan

p = 2 → Euclidean
```

---

## Uniform Weights

All neighbors have equal influence.

```text
weights="uniform"
```

---

## Distance Weights

Closer neighbors have greater influence.

```text
weights="distance"
```

### Interview Question

**Why might we use distance weighting?**

Answer:

> "Distance weighting gives more influence to closer neighbors because they are generally more similar to the query point."

---

# 20. Cross-Validation

Cross-validation helps us estimate how well a model performs on unseen data.

For example:

```text
5-Fold Cross-Validation
```

The training data is divided into 5 parts.

```text
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
```

Each fold gets a chance to act as validation data.

Then we calculate the average score.

### Why use it in KNN?

Because KNN performance depends heavily on:

```text
K
weights
distance metric
```

We can tune these parameters using:

```python
GridSearchCV
```

---

# 21. KNN vs Logistic Regression

| Feature             | KNN                     | Logistic Regression       |
| ------------------- | ----------------------- | ------------------------- |
| Type                | Supervised              | Supervised                |
| Classification      | Yes                     | Yes                       |
| Main idea           | Nearest neighbors       | Probability using sigmoid |
| Distance-based      | Yes                     | No                        |
| Feature scaling     | Very important          | Often important           |
| Coefficients        | No                      | Yes                       |
| Training            | Very little computation | Learns parameters         |
| Prediction          | Can be expensive        | Usually fast              |
| Interpretability    | Lower                   | Higher                    |
| Non-linear patterns | Can handle naturally    | Basic form is linear      |

### Main Difference

Logistic Regression:

```text
Features
 ↓
Linear Score
 ↓
Sigmoid
 ↓
Probability
 ↓
Class
```

KNN:

```text
New Point
 ↓
Calculate Distances
 ↓
Find Neighbors
 ↓
Voting
 ↓
Class
```

---

# 22. KNN vs Decision Tree

| Feature             | KNN               | Decision Tree        |
| ------------------- | ----------------- | -------------------- |
| Main idea           | Nearest neighbors | If-else splits       |
| Distance-based      | Yes               | No                   |
| Feature scaling     | Important         | Usually not required |
| Model structure     | Stores examples   | Learns tree          |
| Prediction          | Distance + voting | Follow decision path |
| Interpretability    | Lower             | High                 |
| Non-linear patterns | Yes               | Yes                  |
| Prediction speed    | Can be slower     | Usually fast         |

### Example Decision Tree

```text
Credit Score > 700?
       |
      Yes
       ↓
Salary > 50000?
       |
      Yes
       ↓
Loan Approved
```

KNN does not create these explicit rules.

---

# 23. Advantages of KNN

### 1. Simple to Understand

The basic idea is easy:

```text
Find nearby points
→ Vote
→ Predict
```

### 2. Easy to Implement

Scikit-learn provides:

```python
KNeighborsClassifier
```

### 3. Can Handle Non-Linear Patterns

KNN does not require a linear decision boundary.

### 4. Flexible

Different:

```text
K
distance metrics
weights
```

can be used.

### 5. Little Training

KNN does not learn a large set of coefficients during fitting.

---

# 24. Disadvantages of KNN

### 1. Prediction Can Be Slow

KNN may need to calculate distances to many training points.

### 2. Sensitive to Feature Scaling

Different feature ranges can distort distance.

### 3. Sensitive to K

Bad K selection can cause:

```text
Overfitting
or
Underfitting
```

### 4. Sensitive to Irrelevant Features

Unhelpful features can affect distance calculations.

### 5. Sensitive to High Dimensions

With many features, distance-based methods can become less effective. This is related to the **curse of dimensionality**.

### 6. Stores Training Data

KNN needs access to the training examples during prediction.

---

# 25. When to Use KNN

KNN can be useful when:

* Dataset is relatively small
* Similar examples tend to have similar labels
* Decision boundary may be non-linear
* Interpretability of local neighbors is useful
* Distance between examples is meaningful

### Example

```text
Customer similarity
Recommendation
Pattern recognition
Small classification datasets
```

---

# 26. When Not to Use KNN

KNN may be less suitable when:

* Dataset is extremely large
* Prediction speed is very important
* There are many irrelevant features
* Feature scaling is difficult
* Data has very high dimensionality
* Distance does not represent similarity well

For large datasets, prediction can become computationally expensive because many training points may need to be considered.

---

# 27. Important Interview Questions

## Q1. What is KNN?

**Answer:**

> "KNN stands for K-Nearest Neighbors. It is a supervised machine learning algorithm that predicts the class of a new data point based on the majority class of its K nearest training points."

---

## Q2. What does K mean?

**Answer:**

> "K represents the number of nearest neighbors considered for prediction."

---

## Q3. Why is feature scaling important in KNN?

**Answer:**

> "Because KNN uses distance calculations. Features with larger numerical ranges can dominate the distance if we don't scale them."

---

## Q4. What happens if K is too small?

**Answer:**

> "The model can become sensitive to noise and individual data points, increasing the risk of overfitting."

---

## Q5. What happens if K is too large?

**Answer:**

> "The model can become too smooth and may underfit the data."

---

## Q6. What is Euclidean distance?

**Answer:**

> "Euclidean distance is the straight-line distance between two points."

---

## Q7. What is `weights="distance"`?

**Answer:**

> "It gives more influence to closer neighbors and less influence to farther neighbors."

---

## Q8. Does KNN learn coefficients?

**Answer:**

> "No. KNN does not learn coefficients like Logistic Regression. It stores the training examples and uses distances to make predictions."

---

## Q9. Is KNN supervised or unsupervised?

**Answer:**

> "KNN is a supervised learning algorithm because it learns from labeled training data."

---

## Q10. Why do we use cross-validation?

**Answer:**

> "Cross-validation helps us evaluate different K values and other parameters more reliably and helps reduce dependence on one particular train-validation split."

---

# 28. Common Interview Trap

### Trap 1

**Interviewer: Does KNN have a sigmoid function?**

Answer:

> "No. KNN does not use a sigmoid function. It uses distance calculations and neighbor voting."

---

### Trap 2

**Interviewer: Does KNN learn coefficients?**

Answer:

> "No. KNN does not learn coefficients. It stores training examples and uses them during prediction."

---

### Trap 3

**Interviewer: Is scaling necessary for KNN?**

Answer:

> "Feature scaling is very important because KNN uses distance calculations."

---

### Trap 4

**Interviewer: Is K = 1 always best?**

Answer:

> "No. K = 1 can be sensitive to noise. The best K should be selected using validation or cross-validation."

---

### Trap 5

**Interviewer: Does KNN use a 0.5 threshold by default?**

Answer:

> "No. Standard KNN classification uses neighbor voting. `predict_proba()` can provide class probabilities, but the basic KNN decision is based on the neighbors."

---

# 29. Loan Approval Project Explanation

If the interviewer asks:

### "Explain your KNN Loan Approval project."

You can say:

> "I used KNN classification to predict whether a loan would be approved or not. My dataset contains features such as age, salary, years of experience, credit score, and loan amount. The target variable is loan approval, where 0 represents not approved and 1 represents approved.
>
> Since KNN is distance-based, I used StandardScaler to scale the numerical features. I split the data into training and testing sets and used cross-validation to compare different values of K along with distance metrics and weighting methods. The model predicts a new customer's loan status by finding the nearest training customers and using their class information.
>
> Finally, I evaluated the model using accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix."

### If interviewer asks:

**"Why did you use KNN?"**

Answer:

> "I wanted to understand a distance-based classification approach and compare predictions based on similarity between customers."

### If interviewer asks:

**"What was the most important preprocessing step?"**

Answer:

> "Feature scaling, because KNN depends on distance."

---

# 30. Quick Revision

## KNN in one line

> **KNN predicts a new data point using the classes of its nearest neighbors.**

### Remember:

```text
KNN
 ↓
Distance-Based
 ↓
Find Neighbors
 ↓
Choose K
 ↓
Voting
 ↓
Prediction
```

### Important Parameters

```text
n_neighbors
weights
metric
p
algorithm
```

### Important preprocessing

```text
StandardScaler
```

### Important evaluation

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Confusion Matrix
```

### Important tuning

```text
K
weights
metric
```

### Small K

```text
Overfitting Risk ↑
```

### Large K

```text
Underfitting Risk ↑
```

### Most important interview sentence

> **"KNN is a distance-based supervised learning algorithm that predicts a data point based on the majority class of its K nearest neighbors, so feature scaling is very important."**

---

# Remember This

```text
KNN = K-Nearest Neighbors

Supervised Learning
       ↓
Distance-Based
       ↓
Scale Features
       ↓
Choose K
       ↓
Calculate Distance
       ↓
Find K Neighbors
       ↓
Majority Voting
       ↓
Prediction
```

**KNN does not learn coefficients.**

**KNN does not use sigmoid.**

**KNN does not normally use a fixed 0.5 threshold for its basic classification decision.**

**Feature scaling is very important.**

**Small K → overfitting risk**

**Large K → underfitting risk**

**Use cross-validation to help choose K.**
