# Decision Tree Regression

## 1. Algorithm Overview

**Decision Tree Regression** is a **supervised machine learning algorithm** used to predict a **continuous numerical target**.

Instead of fitting a straight line, a Decision Tree divides the dataset into smaller groups using a sequence of **if-else conditions**.

### Example

Suppose we want to predict house price using area.

The tree may learn rules such as:

```text
Is Area <= 1000?
       ↓
     Yes
       ↓
  Predict ₹45 Lakh

       No
       ↓
Is Area <= 1500?
       ↓
     Yes
       ↓
  Predict ₹65 Lakh

       No
       ↓
  Predict ₹90 Lakh
```

### Main Idea

```text
Dataset
   ↓
Find Best Split
   ↓
Split Data
   ↓
Repeat Splitting
   ↓
Create Tree
   ↓
Predict
```

Decision Tree Regression is useful when the relationship between features and target is **nonlinear**.

---

# 2. Core Intuition

A Decision Tree works like a sequence of questions.

For example, for house-price prediction:

```text
                Area <= 1000?
                /           \
              Yes            No
              ↓              ↓
        Bedrooms <= 2?    Area <= 1500?
          /      \          /       \
        Yes      No       Yes       No
        ↓         ↓        ↓         ↓
      40L       50L      65L       90L
```

Each internal node contains a **decision condition**.

Each branch represents an outcome of that condition.

Each leaf contains the final prediction.

### Important

Decision Trees do not require a straight-line relationship between features and target.

They can learn complex nonlinear patterns.

---

# 3. Decision Tree Structure

A Decision Tree contains:

### Root Node

The first decision in the tree.

```text
        Area <= 1200?
```

### Internal Node

A decision made after the root.

```text
Bedrooms <= 3?
```

### Branch

The result of a decision.

```text
Yes
No
```

### Leaf Node

The final prediction.

```text
₹65 Lakh
```

### Complete Structure

```text
             Root
              ↓
        Internal Node
          /        \
       Branch     Branch
         ↓          ↓
      Internal    Leaf
         ↓       Prediction
        Leaf
     Prediction
```

---

# 4. How Decision Tree Regression Works

The general process is:

```text
Dataset
   ↓
Select Features and Target
   ↓
Train-Test Split
   ↓
Find Best Feature and Split
   ↓
Create Decision Node
   ↓
Split Dataset
   ↓
Repeat Recursively
   ↓
Reach Leaf Nodes
   ↓
Predict
   ↓
Evaluate
```

The tree repeatedly divides the data into smaller groups.

The goal is to create groups where the target values are as similar as possible.

---

# 5. Splitting in Decision Tree Regression

For regression, the tree searches for splits that reduce the **impurity/error** within the resulting groups.

A common measure is **Mean Squared Error (MSE)**.

For a node:

```math
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\bar{y})^2
```

Where:

* `yᵢ` = actual target value
* `ȳ` = mean target value in the node
* `n` = number of samples

The algorithm tries different:

```text
Feature
   +
Threshold
```

and chooses a split that produces the best reduction in error.

---

# 6. Example of a Split

Suppose we have:

| Area | Price |
| ---: | ----: |
|  800 |    35 |
|  900 |    40 |
| 1100 |    48 |
| 1300 |    60 |
| 1500 |    72 |

The tree may test:

```text
Area <= 1000
```

This creates:

```text
Left Node:

800 → 35
900 → 40

Right Node:

1100 → 48
1300 → 60
1500 → 72
```

The tree evaluates whether this split reduces the prediction error.

It continues splitting if further splitting improves the objective.

---

# 7. Prediction in Decision Tree Regression

At a leaf node, the prediction is generally the **mean target value of the training samples reaching that leaf**.

Example:

```text
Leaf contains:

₹40 Lakh
₹50 Lakh
₹60 Lakh
```

Prediction:

```text
(40 + 50 + 60) / 3
= ₹50 Lakh
```

Therefore:

```text
New House
    ↓
Follow tree conditions
    ↓
Reach leaf
    ↓
Return leaf prediction
```

---

# 8. Python Implementation

Scikit-learn provides:

```python
from sklearn.tree import DecisionTreeRegressor
```

### Basic Code

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("data.csv")

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 9. Model Evaluation

Decision Tree Regression uses standard regression metrics.

### R² Score

```python
print("Testing R² Score:", model.score(X_test, y_test))
```

If you want percentage form:

```python
print("Testing R² Score:", model.score(X_test, y_test) * 100)
```

### Important

For regression:

```text
model.score()
```

returns **R²**, not classification accuracy.

So prefer:

```text
Testing R² Score
```

instead of:

```text
Testing Accuracy
```

---

# 10. MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
```

Lower MAE is generally better.

---

# 11. MSE

Mean Squared Error gives more importance to large errors.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
```

Lower MSE is generally better.

---

# 12. RMSE

RMSE is the square root of MSE.

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)
```

Lower RMSE is generally better.

---

# 13. Complete Evaluation Code

```python
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

---

# 14. Actual vs Predicted Graph

A useful graph for regression is **Actual vs Predicted**.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red"
)

plt.title("Decision Tree Regression: Actual vs Predicted")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.show()
```

### How to Read It

```text
Points close to diagonal
        ↓
Better predictions

Points far from diagonal
        ↓
Larger prediction errors
```

---

# 15. Residual Analysis

Residual:

```math
Residual = Actual - Predicted
```

Python:

```python
residuals = y_test - y_pred
```

### Residual Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(0)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Decision Tree Regression: Residuals")

plt.show()
```

A useful residual pattern generally has errors distributed around zero without a strong systematic pattern.

---

# 16. Important Hyperparameters

Decision Trees have several important hyperparameters.

The most important for beginners are:

```text
max_depth
min_samples_split
min_samples_leaf
max_features
```

---

# 17. max_depth

`max_depth` controls the maximum depth of the tree.

Example:

```python
model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)
```

### Small Depth

```text
Small tree
   ↓
Simpler model
   ↓
May underfit
```

### Large Depth

```text
Large tree
   ↓
More complex model
   ↓
Higher overfitting risk
```

---

# 18. min_samples_split

`min_samples_split` specifies the minimum number of samples required to split an internal node.

Example:

```python
model = DecisionTreeRegressor(
    min_samples_split=10,
    random_state=42
)
```

If a node has fewer than 10 samples, it cannot be split.

Increasing this value can make the tree less complex.

---

# 19. min_samples_leaf

`min_samples_leaf` specifies the minimum number of samples required in a leaf node.

Example:

```python
model = DecisionTreeRegressor(
    min_samples_leaf=5,
    random_state=42
)
```

This prevents the tree from creating extremely small leaves.

It can help reduce overfitting.

---

# 20. Overfitting

Decision Trees are highly prone to overfitting if allowed to grow too deeply.

Example:

```python
model = DecisionTreeRegressor(
    max_depth=None,
    random_state=42
)
```

The tree may become very complex.

It can learn noise from the training data.

### Typical Pattern

```text
Training R² → Very High

Testing R² → Much Lower
```

This indicates possible overfitting.

---

# 21. Underfitting

Underfitting occurs when the tree is too simple to capture important patterns.

Example:

```python
model = DecisionTreeRegressor(
    max_depth=2,
    random_state=42
)
```

The tree may not have enough complexity.

### Typical Pattern

```text
Training R² → Low

Testing R² → Low
```

Possible solution:

```text
Increase max_depth
```

But always validate the result on unseen data.

---

# 22. Controlling Overfitting

You can control tree complexity using:

```python
DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

The exact values depend on the dataset.

Do not blindly use the same values for every project.

---

# 23. Feature Scaling

Decision Tree Regression generally **does not require feature scaling**.

For example:

```text
Area = 1500
Bedrooms = 3
Bathrooms = 2
```

A Decision Tree mainly compares values against thresholds.

Therefore, StandardScaler is usually unnecessary.

Unlike algorithms such as:

```text
Linear Regression with regularization
KNN
SVM
Neural Networks
```

Decision Trees are generally not sensitive to feature magnitude.

---

# 24. Categorical Features

Decision Trees in scikit-learn generally require numerical input.

Categorical features can therefore be encoded.

For example:

```text
Area Type
---------
Built-up
Plot
Super built-up
```

One-hot encoding can be used:

```python
from sklearn.preprocessing import OneHotEncoder
```

For a complete preprocessing workflow, `ColumnTransformer` and `Pipeline` are useful.

---

# 25. Decision Tree Regression With Multiple Features

Decision Tree Regression can use multiple features.

Example:

```python
X = df[
    [
        "area",
        "bedrooms",
        "bathrooms",
        "balcony"
    ]
]

y = df["price"]
```

The tree can learn rules such as:

```text
Area <= 1200?
        ↓
Bedrooms <= 2?
        ↓
Bathrooms <= 2?
        ↓
Predict Price
```

This allows the model to capture interactions between features automatically.

---

# 26. Visualizing the Decision Tree

Scikit-learn provides a way to visualize the tree.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)

plt.show()
```

This can help you understand how the model makes decisions.

---

# 27. Decision Tree Depth Experiment

A useful experiment is to compare different depths.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

for depth in range(1, 11):

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(
        "Depth:", depth,
        "R²:", r2_score(y_test, y_pred)
    )
```

You can compare:

```text
Depth 1 → R²
Depth 2 → R²
Depth 3 → R²
...
Depth 10 → R²
```

Choose a depth based on validation/generalization performance.

---

# 28. Cross-Validation

Cross-validation provides a more reliable estimate of model performance.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("CV R²:", scores.mean())
```

Cross-validation can also help compare different tree depths.

---

# 29. Advantages

1. Easy to understand.
2. Easy to visualize.
3. Can model nonlinear relationships.
4. Does not require feature scaling.
5. Can capture feature interactions.
6. Works with multiple features.
7. Can model complex decision boundaries.
8. Requires relatively little preprocessing for numerical data.
9. Useful as a baseline for tree-based models.

---

# 30. Disadvantages

1. Can easily overfit.
2. Small data changes can produce a different tree.
3. A single tree may have lower generalization performance than ensembles.
4. Deep trees can become very complex.
5. Predictions are piecewise constant.
6. Not ideal for smooth relationships.
7. Hyperparameter tuning may be necessary.
8. Extrapolation beyond the training range is limited.

---

# 31. Decision Tree vs Linear Regression

| Feature            | Linear Regression    | Decision Tree         |
| ------------------ | -------------------- | --------------------- |
| Relationship       | Linear               | Nonlinear             |
| Model structure    | Equation             | Tree                  |
| Feature scaling    | Usually not required | Not required          |
| Interpretability   | High                 | High                  |
| Overfitting risk   | Lower                | Higher                |
| Nonlinear patterns | Poor                 | Good                  |
| Interactions       | Must be added        | Learned automatically |
| Prediction shape   | Smooth line/plane    | Piecewise constant    |

---

# 32. Decision Tree vs Polynomial Regression

| Feature                   | Polynomial Regression        | Decision Tree             |
| ------------------------- | ---------------------------- | ------------------------- |
| Relationship              | Curved polynomial            | Rule-based                |
| Feature transformation    | Required                     | Not required              |
| Scaling                   | Sometimes useful             | Usually unnecessary       |
| Nonlinear patterns        | Good for polynomial patterns | Good for complex patterns |
| Overfitting risk          | Higher with degree           | Higher with depth         |
| Interpretability          | Medium                       | High                      |
| Main complexity parameter | Degree                       | Depth                     |

---

# 33. Decision Tree vs Random Forest

| Feature             | Decision Tree | Random Forest         |
| ------------------- | ------------- | --------------------- |
| Number of trees     | One           | Many                  |
| Overfitting         | Higher        | Usually lower         |
| Stability           | Lower         | Higher                |
| Training complexity | Lower         | Higher                |
| Prediction          | One tree      | Average of many trees |
| Accuracy            | Often lower   | Often better          |
| Interpretability    | Easier        | More difficult        |

Random Forest is an **ensemble of Decision Trees**.

---

# 34. Decision Tree vs XGBoost

| Feature             | Decision Tree   | XGBoost                        |
| ------------------- | --------------- | ------------------------------ |
| Model type          | Single tree     | Boosted trees                  |
| Complexity          | Lower           | Higher                         |
| Training            | Simpler         | More complex                   |
| Overfitting control | Tree parameters | Many regularization parameters |
| Performance         | Good baseline   | Often very strong              |
| Interpretability    | Easier          | More difficult                 |

The best algorithm depends on the dataset and validation performance.

---

# 35. When to Use Decision Tree Regression

Use Decision Tree Regression when:

* The target is continuous.
* The relationship is nonlinear.
* Feature interactions are important.
* You want a rule-based model.
* You do not want to perform feature scaling.
* Interpretability is useful.
* You need a simple tree-based baseline.

### Example Applications

```text
House Price Prediction
Sales Prediction
Demand Prediction
Energy Consumption
Vehicle Price Prediction
```

---

# 36. When NOT to Use a Single Decision Tree

A single Decision Tree may not be the best choice when:

* Very high predictive performance is required.
* The tree strongly overfits.
* The relationship is smooth and approximately linear.
* Small changes in data cause unstable predictions.
* Ensemble models perform substantially better.

In such cases, consider:

```text
Random Forest
Gradient Boosting
XGBoost
LightGBM
```

---

# 37. Common Mistake: Very Deep Tree

A common beginner mistake is:

```python
DecisionTreeRegressor(
    max_depth=None
)
```

without checking overfitting.

A very deep tree can memorize training observations.

### Better Approach

```text
Try different depths
       ↓
Evaluate validation performance
       ↓
Compare Training vs Testing
       ↓
Check overfitting
       ↓
Choose suitable depth
```

---

# 38. Practical Experiment

For your **AI Forge** repository, practice Decision Tree Regression in this order:

```text
1. Load dataset
        ↓
2. Select features and target
        ↓
3. Train-test split
        ↓
4. Train Decision Tree
        ↓
5. Predict
        ↓
6. Calculate MAE / RMSE / R²
        ↓
7. Plot Actual vs Predicted
        ↓
8. Plot Residuals
        ↓
9. Visualize the tree
        ↓
10. Try max_depth = 1–10
        ↓
11. Compare Training vs Testing R²
        ↓
12. Check overfitting
        ↓
13. Try Random Forest
```

---

# 39. Interview Questions

## Basic

1. What is Decision Tree Regression?
2. Is Decision Tree Regression supervised or unsupervised?
3. Is Decision Tree Regression classification or regression?
4. What is a root node?
5. What is an internal node?
6. What is a leaf node?
7. How does a Decision Tree make predictions?
8. What is a split?
9. What is a threshold?
10. What is `max_depth`?

## Technical

11. How does a Decision Tree choose the best split?
12. What is MSE in Decision Tree Regression?
13. Why can Decision Trees overfit?
14. How can you control overfitting?
15. What is `min_samples_split`?
16. What is `min_samples_leaf`?
17. Does Decision Tree require feature scaling?
18. How does a regression tree calculate the leaf prediction?
19. Why are Decision Trees unstable?
20. What is pruning?
21. How is Decision Tree different from Random Forest?
22. How do you select `max_depth`?
23. Why use cross-validation?
24. Can Decision Trees capture nonlinear relationships?
25. Can Decision Trees capture feature interactions?

---

# 40. Common Interview Traps

### Trap 1: Decision Trees require feature scaling

**False.**

Decision Trees generally do not require feature scaling because they use threshold-based splits.

---

### Trap 2: A deeper tree is always better

**False.**

A deeper tree is more flexible but has a higher risk of overfitting.

---

### Trap 3: Decision Tree Regression predicts using a straight line

**False.**

A regression tree creates regions and generally predicts a constant value for each leaf.

---

### Trap 4: Decision Trees cannot model nonlinear relationships

**False.**

Decision Trees naturally model nonlinear relationships.

---

### Trap 5: `model.score()` gives accuracy

**False for regression.**

For `DecisionTreeRegressor`, `model.score(X_test, y_test)` returns **R²**.

---

### Trap 6: More tree depth always improves testing performance

**False.**

Increasing depth can improve training performance while reducing generalization.

---

### Trap 7: Decision Trees are always stable

**False.**

Small changes in training data can sometimes produce substantially different tree structures.

---

# 41. Project-Based Interview Question

### Why would you use Decision Tree Regression in a house-price project?

**Answer:**

> “I would use Decision Tree Regression because house prices can have nonlinear relationships with features such as area, BHK, and bathrooms. A Decision Tree can automatically learn threshold-based rules and feature interactions without requiring feature scaling. I would tune parameters such as max_depth and compare the model using R², MAE, and RMSE to control overfitting.”

---

# 42. One-Minute Interview Explanation

> **“Decision Tree Regression is a supervised machine learning algorithm used to predict continuous numerical values. It works by recursively splitting the dataset using feature-based conditions and creating a tree structure. For regression, the prediction at a leaf is generally based on the average target value of the training samples reaching that leaf. Decision Trees can capture nonlinear relationships and feature interactions and usually do not require feature scaling. However, deep trees can overfit, so I control complexity using parameters such as max_depth, min_samples_split, and min_samples_leaf. I evaluate the model using metrics such as MAE, RMSE, and R².”**

---

# 43. Quick Revision

```text
Decision Tree Regression
        ↓
Supervised Learning
        ↓
Regression
        ↓
Continuous Target
        ↓
Find Best Split
        ↓
Create Tree
        ↓
Reach Leaf
        ↓
Predict Leaf Value
        ↓
MAE / RMSE / R²
```

### Important Parameters

```text
max_depth
min_samples_split
min_samples_leaf
max_features
```

### Most Important Concept

```text
More Depth
    ↓
More Complexity
    ↓
Higher Overfitting Risk
```

---

# 44. Recommended Practice

For your **AI Forge** repository:

```text
01. Train a Decision Tree with one feature
        ↓
02. Calculate R²
        ↓
03. Calculate MAE
        ↓
04. Calculate RMSE
        ↓
05. Plot Actual vs Predicted
        ↓
06. Plot Residuals
        ↓
07. Visualize the tree
        ↓
08. Try different max_depth values
        ↓
09. Compare Training R² and Testing R²
        ↓
10. Use Cross-Validation
        ↓
11. Compare Decision Tree vs Random Forest
```

---

# 45. Final Takeaway

Decision Tree Regression is a powerful and easy-to-understand tree-based regression algorithm.

Remember these five points:

```text
1. Decision Trees use if-else style splits.

2. They can learn nonlinear relationships.

3. Regression leaves generally predict an average target value.

4. Feature scaling is usually not required.

5. Very deep trees can overfit.
```

The most important practical workflow is:

```text
Prepare Data
     ↓
Train-Test Split
     ↓
Train Decision Tree
     ↓
Predict
     ↓
Evaluate
     ↓
Check Training vs Testing
     ↓
Tune max_depth
     ↓
Check Overfitting
     ↓
Compare with Random Forest
```

### Learning Path

```text
01_Linear_Regression
        ↓
02_Multiple_Linear_Regression
        ↓
03_Polynomial_Regression
        ↓
04_Decision_Tree_Regression
        ↓
05_Random_Forest_Regression
        ↓
06_Ridge_Regression
        ↓
07_Lasso_Regression
        ↓
08_ElasticNet
        ↓
09_XGBoost_Regression
```

### Final Rule

> **Do not choose a Decision Tree because it gets a very high training R². Choose the model based on how well it generalizes to unseen data.**
