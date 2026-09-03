# Polynomial Regression

## 1. Algorithm Overview

**Polynomial Regression** is a **supervised machine learning algorithm** used to predict a **continuous numerical target** when the relationship between the input features and target is not adequately represented by a straight line.

It extends Linear Regression by creating **polynomial features** such as:

* `x²`
* `x³`
* `x⁴`

The model can therefore learn a **curved relationship** between the input and target.

### Example

Suppose we want to predict salary based on experience.

A simple Linear Regression model assumes:

```text
Salary = b₀ + b₁(Experience)
```

Polynomial Regression can model:

```text
Salary = b₀ + b₁(Experience) + b₂(Experience²)
```

### Main Idea

```text
Original Feature
      ↓
Polynomial Features
      ↓
Linear Regression
      ↓
Prediction
```

Polynomial Regression is especially useful when a scatter plot shows a clear **curved relationship**.

---

# 2. Core Intuition

Linear Regression tries to fit a straight line:

```text
y
│          •
│       •
│    •
│  •
│ •
└──────────────── x
```

But some datasets have relationships like:

```text
y
│       •
│     •   •
│   •       •
│ •           •
│
└──────────────── x
```

A straight line may underfit this relationship.

Polynomial Regression introduces powers of the features so that the model can fit a curve.

### Example

```text
Experience → Salary

Linear Regression:
Salary = b₀ + b₁x

Polynomial Regression:
Salary = b₀ + b₁x + b₂x²
```

---

# 3. Linear vs Polynomial Regression

## Linear Regression

Uses the original feature:

```math
ŷ = b₀ + b₁x
```

Produces a straight relationship.

## Polynomial Regression

Uses transformed features:

```math
ŷ = b₀ + b₁x + b₂x² + b₃x³ + ...
```

Produces a curved relationship.

### Important

Polynomial Regression is still considered a **linear model in its parameters** because the coefficients `b₀, b₁, b₂...` are linear.

The relationship with the original input `x` can be nonlinear.

---

# 4. Mathematical Foundation

The general polynomial equation is:

```math
ŷ = b₀ + b₁x + b₂x² + b₃x³ + ... + b_dx^d
```

Where:

* `ŷ` = predicted target
* `b₀` = intercept
* `b₁, b₂, ...` = coefficients
* `x` = input feature
* `d` = polynomial degree

### Degree 1

```math
ŷ = b₀ + b₁x
```

Equivalent to Linear Regression.

### Degree 2

```math
ŷ = b₀ + b₁x + b₂x²
```

Quadratic curve.

### Degree 3

```math
ŷ = b₀ + b₁x + b₂x² + b₃x³
```

Cubic curve.

---

# 5. Polynomial Degree

The **degree** controls the complexity of the polynomial.

```text
Degree 1 → Straight line
Degree 2 → Simple curve
Degree 3 → More flexible curve
Degree 4+ → Increasingly complex
```

Example:

```python
PolynomialFeatures(degree=2)
```

creates:

```text
x
x²
```

For:

```python
PolynomialFeatures(degree=3)
```

it creates:

```text
x
x²
x³
```

### Important

A higher degree does **not automatically mean a better model**.

Too high a degree can cause **overfitting**.

---

# 6. How Polynomial Regression Works

The general process is:

```text
Dataset
   ↓
Select Features and Target
   ↓
Train-Test Split
   ↓
Create Polynomial Features
   ↓
Train Linear Regression
   ↓
Predict
   ↓
Evaluate
```

The important point is that Polynomial Regression usually works by:

1. Transforming the original features.
2. Applying Linear Regression to the transformed features.

---

# 7. PolynomialFeatures

Scikit-learn provides:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Example:

```python
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)
```

If:

```text
X = [2, 3]
```

degree 2 produces polynomial terms such as:

```text
1
x
x²
```

The `1` represents the intercept term.

---

# 8. Python Implementation

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)
```

### Important

Use:

```python
fit_transform()
```

on the training data.

Use:

```python
transform()
```

on the test data.

Do **not** fit the polynomial transformer separately on the test set.

---

# 9. Model Evaluation

Polynomial Regression can be evaluated using the same regression metrics used for Linear Regression.

### MAE

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
```

Measures the average absolute prediction error.

---

### MSE

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
```

MSE gives more importance to large errors.

---

### RMSE

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)
```

Lower RMSE is generally better.

---

### R² Score

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R²:", r2)
```

R² measures how much variation in the target is explained by the model.

### Important

```text
R² is NOT accuracy.
```

---

# 10. Best Graph for Polynomial Regression

For a **single feature**, the most useful visualization is:

**Actual data points + polynomial curve**

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=X["area"], y=y)

plt.plot(
    X_sorted,
    model.predict(poly.transform(X_sorted))
)

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Polynomial Regression")

plt.show()
```

### Why sort X?

If `X` is not sorted, the plotted prediction line can jump back and forth.

Use:

```python
X_sorted = X.sort_values("area")
```

before plotting.

---

# 11. Actual vs Predicted Graph

Another useful graph is:

```python
sns.scatterplot(x=y_test, y=y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()
```

Points closer to the diagonal relationship generally indicate better predictions.

---

# 12. Residual Analysis

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
sns.scatterplot(x=y_pred, y=residuals)

plt.axhline(0)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")

plt.show()
```

### Good residual pattern

Residuals should generally be:

* Randomly distributed
* Around zero
* Without a strong pattern
* With reasonably consistent spread

A clear pattern can indicate that the model is not capturing the relationship properly.

---

# 13. Overfitting

Polynomial Regression is especially sensitive to overfitting.

For example:

```text
Degree = 2
→ Simple curve

Degree = 3
→ More flexible

Degree = 10
→ Very flexible
```

A very high degree can make the model follow noise in the training data.

### Typical pattern

```text
Training R² → Very High
Testing R²  → Much Lower
```

This is a common sign of overfitting.

---

# 14. Underfitting

Underfitting happens when the polynomial degree is too low to represent the relationship.

Example:

```text
Actual relationship → strongly curved

Degree 1 → too simple
```

The model may fail to capture the important pattern.

Possible solution:

```text
Increase polynomial degree
```

But increase it carefully and validate on unseen data.

---

# 15. Choosing the Degree

Do not select the degree only because it gives the highest training score.

Instead, compare different degrees using validation data or cross-validation.

Example:

```python
from sklearn.metrics import r2_score

for degree in range(1, 6):

    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    print(
        degree,
        r2_score(y_test, y_pred)
    )
```

You can compare:

```text
Degree 1 → R²
Degree 2 → R²
Degree 3 → R²
Degree 4 → R²
Degree 5 → R²
```

Choose the degree based on **validation/generalization performance**, not simply training performance.

---

# 16. Cross-Validation

Cross-validation gives a more reliable estimate of model performance.

Example:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train_poly,
    y_train,
    cv=5,
    scoring="r2"
)

print("CV R²:", scores.mean())
```

### Why use Cross-Validation?

It helps determine whether the chosen polynomial degree generalizes well to unseen data.

---

# 17. Feature Scaling

Polynomial features can become very large.

For example:

```text
x = 100

x² = 10,000

x³ = 1,000,000
```

With higher degrees, feature magnitudes can grow quickly.

Scaling can therefore be useful, especially with higher-degree polynomial features or regularized models.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)
```

---

# 18. Polynomial Regression With Multiple Features

Polynomial Regression can also be used with multiple features.

Suppose:

```text
X₁ = Area
X₂ = Bedrooms
```

Degree 2 can create terms such as:

```text
Area
Bedrooms
Area²
Bedrooms²
Area × Bedrooms
```

The model can therefore capture:

* Curvature
* Feature interactions

### Example

```python
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

For multiple features, the number of generated features can grow quickly.

This can increase:

* Computation
* Memory usage
* Overfitting risk

---

# 19. Interaction Features

PolynomialFeatures can create interaction terms.

For example:

```text
Area × Bedrooms
```

can represent an interaction between the two features.

This allows the model to represent relationships that depend on combinations of variables.

---

# 20. Regularization

Polynomial Regression with many features can overfit.

Regularization can help control this.

Common choices:

### Ridge Regression

Uses L2 regularization.

```text
Ridge + Polynomial Features
```

### Lasso Regression

Uses L1 regularization.

```text
Lasso + Polynomial Features
```

A common pipeline is:

```text
Polynomial Features
        ↓
Scaling
        ↓
Ridge Regression
```

This is often useful when polynomial feature expansion creates many correlated features.

---

# 21. Pipeline

A pipeline keeps preprocessing and modeling together.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

Pipelines also help reduce the risk of inconsistent preprocessing.

---

# 22. Data Leakage

Avoid fitting transformations on the complete dataset before splitting.

Incorrect:

```python
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly,
    y,
    test_size=0.2,
    random_state=42
)
```

Better:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

The test set should remain unseen during training and preprocessing fitting.

---

# 23. Advantages

1. Can model curved relationships.
2. Simple extension of Linear Regression.
3. Easy to implement using scikit-learn.
4. Can capture nonlinear patterns.
5. Can model feature interactions.
6. More interpretable than many complex nonlinear models.
7. Useful when the relationship has a known smooth polynomial shape.

---

# 24. Disadvantages

1. High-degree polynomials can overfit.
2. Number of features can grow quickly.
3. Higher-degree terms can become numerically large.
4. Sensitive to outliers.
5. Coefficient interpretation becomes difficult.
6. Performance can become unstable outside the training range.
7. Not suitable for every nonlinear relationship.
8. Polynomial feature expansion can increase computational cost.

---

# 25. Polynomial Regression vs Linear Regression

| Feature          | Linear Regression    | Polynomial Regression |
| ---------------- | -------------------- | --------------------- |
| Relationship     | Approximately linear | Curved/nonlinear      |
| Features         | Original features    | Polynomial features   |
| Degree           | 1                    | 2 or higher           |
| Complexity       | Low                  | Higher                |
| Overfitting risk | Lower                | Higher                |
| Interpretability | Easier               | More difficult        |
| Curved patterns  | Poor                 | Better                |
| Training speed   | Very fast            | Usually slower        |

---

# 26. Polynomial Regression vs Other Algorithms

| Algorithm             | Nonlinear Patterns | Interpretability | Overfitting Risk |
| --------------------- | ------------------ | ---------------- | ---------------- |
| Linear Regression     | Low                | High             | Low              |
| Polynomial Regression | Medium/High        | Medium           | Medium/High      |
| Decision Tree         | High               | Medium           | High             |
| Random Forest         | High               | Lower            | Medium           |
| XGBoost               | Very High          | Lower            | Medium           |

The best algorithm depends on the dataset and validation performance.

---

# 27. When to Use Polynomial Regression

Use Polynomial Regression when:

* The target is continuous.
* The relationship appears curved.
* Linear Regression underfits.
* The dataset is not extremely high-dimensional.
* You need a relatively simple nonlinear model.
* You want to understand the effect of polynomial terms.

### Example

```text
Temperature → Energy Consumption
Experience → Salary
Speed → Fuel Consumption
Advertising Spend → Sales
```

---

# 28. When NOT to Use Polynomial Regression

Avoid using very high-degree Polynomial Regression when:

* The dataset is very high-dimensional.
* The relationship is extremely complex.
* There is significant noise.
* The model is strongly overfitting.
* Tree-based or boosting models perform substantially better.
* Extrapolation is required far outside the training range.

---

# 29. Common Mistake: Very High Degree

A common beginner mistake is:

```python
PolynomialFeatures(degree=20)
```

just because a high degree produces a very high training score.

This can create a model that memorizes the training data.

### Better approach

```text
Try reasonable degrees
        ↓
Evaluate on validation/test data
        ↓
Compare performance
        ↓
Choose a degree that generalizes well
```

---

# 30. Practical Experiment

For practice, compare:

```text
Degree 1
Degree 2
Degree 3
Degree 4
Degree 5
```

Record:

| Degree | MAE | RMSE | R² |
| -----: | --: | ---: | -: |
|      1 |   - |    - |  - |
|      2 |   - |    - |  - |
|      3 |   - |    - |  - |
|      4 |   - |    - |  - |
|      5 |   - |    - |  - |

Then identify:

* Best validation R²
* Lowest RMSE
* Lowest MAE
* Whether overfitting occurs

---

# 31. Interview Questions

### Basic

1. What is Polynomial Regression?
2. Why do we use Polynomial Regression?
3. How is it different from Linear Regression?
4. What is polynomial degree?
5. What happens when degree increases?
6. What is `PolynomialFeatures`?
7. Is Polynomial Regression linear or nonlinear?

### Technical

8. Why can Polynomial Regression overfit?
9. How do you choose the polynomial degree?
10. Why should we use `fit_transform()` only on training data?
11. What are interaction terms?
12. Why can polynomial features become large?
13. When would you use Ridge with Polynomial Regression?
14. How do you evaluate Polynomial Regression?
15. Why is cross-validation useful?

---

# 32. Common Interview Traps

### Trap 1: Polynomial Regression is completely nonlinear

Not exactly.

It is **nonlinear with respect to the original input**, but linear in its coefficients.

---

### Trap 2: Higher degree is always better

False.

A higher degree increases flexibility but can cause overfitting.

---

### Trap 3: R² = 0.90 means 90% accuracy

False.

R² is an explained-variance-based regression metric, not classification accuracy.

---

### Trap 4: Polynomial Regression only works with one feature

False.

It can work with multiple features and can create interaction terms.

---

### Trap 5: Use `fit_transform()` on test data

Wrong.

Use:

```python
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
```

---

# 33. Project-Based Interview Question

### Why would you use Polynomial Regression in a house-price project?

**Answer:**

> “I would consider Polynomial Regression if the relationship between features such as area and house price is curved and a simple Linear Regression model underfits the data. I would compare different polynomial degrees using validation performance and use regularization if the expanded features cause overfitting.”

---

# 34. One-Minute Interview Explanation

> **“Polynomial Regression is a supervised regression technique that extends Linear Regression by adding polynomial terms such as x² and x³. This allows the model to capture curved relationships between features and a continuous target. In practice, I use PolynomialFeatures to transform the input data and then train a Linear Regression or regularized model on the transformed features. The polynomial degree controls model complexity, so I select it using validation or cross-validation to avoid overfitting.”**

---

# 35. Quick Revision

```text
Polynomial Regression
        ↓
Supervised Learning
        ↓
Regression
        ↓
Continuous Target
        ↓
Add Polynomial Features
        ↓
x, x², x³, ...
        ↓
Linear Regression / Ridge
        ↓
Prediction
        ↓
MAE / RMSE / R²
```

### Key Formula

```math
ŷ = b₀ + b₁x + b₂x² + ... + b_dx^d
```

### Key Concept

```text
Higher Degree
      ↓
More Flexibility
      ↓
Higher Overfitting Risk
```

---

# 36. Recommended Practice

For your **AI Forge** repository, practice Polynomial Regression in this order:

```text
1. Create a simple dataset
        ↓
2. Plot X vs y
        ↓
3. Train Linear Regression
        ↓
4. Check the graph
        ↓
5. Create Polynomial Features
        ↓
6. Train Polynomial Regression
        ↓
7. Plot polynomial curve
        ↓
8. Calculate MAE / RMSE / R²
        ↓
9. Plot residuals
        ↓
10. Compare degree 1–5
        ↓
11. Check overfitting
        ↓
12. Try Ridge + Polynomial Features
```

---

# 37. Final Takeaway

Polynomial Regression is useful when a straight line is too simple but you still want a relatively understandable regression model.

Remember the three most important points:

```text
1. Polynomial Regression adds powers of features.

2. Degree controls model complexity.

3. Higher degree can cause overfitting.
```

The most important practical workflow is:

```text
Linear Regression
       ↓
Check relationship
       ↓
If curved → Polynomial Features
       ↓
Choose degree using validation
       ↓
Evaluate
       ↓
Check residuals
       ↓
Use regularization if necessary
```

### Learning Path

```text
01_Linear_Regression
        ↓
02_Multiple_Linear_Regression
        ↓
03_Polynomial_Regression
        ↓
04_Ridge_Regression
        ↓
05_Lasso_Regression
        ↓
06_ElasticNet
        ↓
07_Decision_Tree_Regression
        ↓
08_Random_Forest_Regression
        ↓
09_XGBoost_Regression
```

**Final rule:** Don't choose Polynomial Regression because the curve looks good on the training data. Choose it because it **generalizes well to unseen data**.
