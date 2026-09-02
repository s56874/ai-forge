# Linear Regression

**Type:** Supervised Learning → Regression
**Main Goal:** Predict a continuous numerical value by learning a linear relationship between input features and a target.

---

## 1. Algorithm Overview

Linear Regression models the relationship between one or more input features and a continuous target.

### Examples

| Input Features       | Target            |
| -------------------- | ----------------- |
| Area, BHK, Bathrooms | House Price       |
| Experience           | Salary            |
| Advertising Spend    | Sales             |
| Temperature          | Electricity Usage |

For multiple features:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

Where:

* `ŷ` = predicted value
* `b₀` = intercept
* `b₁ ... bₙ` = learned coefficients
* `x₁ ... xₙ` = input features

### Main Idea

The model tries to find coefficients that make predictions as close as possible to the actual values.

---

## 2. Core Intuition

Suppose we want to predict house prices using area.

```text
Price
  │
  │              ●
  │          ●
  │       ●
  │    ●
  │ ●
  └──────────────────── Area
```

Linear Regression finds the line that best represents the relationship between `Area` and `Price`.

The line is selected so that the overall prediction error is minimized.

### Important

Linear Regression does **not** require every data point to lie exactly on a straight line.

It finds the best linear approximation to the relationship.

---

## 3. How It Works

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Learn Coefficients
   ↓
Make Predictions
   ↓
Evaluate Model
```

### Step-by-step

### Step 1 — Input data

We have features `X` and target `y`.

```text
X → Features
y → Target
```

### Step 2 — Model assumes a linear relationship

For one feature:

$$
\hat{y}=b_0+b_1x
$$

### Step 3 — Calculate predictions

The model uses its current coefficients to calculate `ŷ`.

### Step 4 — Calculate residuals

$$
Residual = y-\hat{y}
$$

### Step 5 — Minimize error

Ordinary Least Squares chooses coefficients that minimize the sum of squared residuals.

### Step 6 — Use learned coefficients

After training, the model can predict values for unseen data.

---

# 4. Mathematical Foundation

## Simple Linear Regression

With one feature:

$$
\hat{y}=b_0+b_1x
$$

Where:

* `b₀` → intercept
* `b₁` → slope/coefficient
* `x` → input feature
* `ŷ` → prediction

### Example

$$
Price=20+0.05(Area)
$$

For `Area = 1000`:

$$
Price=20+0.05(1000)=70
$$

---

## Multiple Linear Regression

With multiple features:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
$$

Example:

$$
Price =
b_0+
b_1(Area)+
b_2(BHK)+
b_3(Bathrooms)
$$

Each coefficient represents the expected change in prediction for a one-unit increase in that feature, **holding the other included features constant**.

---

## Ordinary Least Squares

Linear Regression commonly uses **Ordinary Least Squares (OLS)**.

It minimizes:

$$
RSS=\sum_{i=1}^{n}(y_i-\hat{y_i})^2
$$

Where:

* `RSS` = Residual Sum of Squares
* `yᵢ` = actual value
* `ŷᵢ` = predicted value

### Why square the residuals?

```text
Positive error + Negative error
            ↓
      could cancel out

Squaring
            ↓
all errors become positive
            ↓
large errors receive more penalty
```

---

## Normal Equation

For OLS, coefficients can be expressed as:

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

This provides a closed-form solution under the usual full-rank conditions.

In practice, libraries such as scikit-learn handle the numerical computation for us.

---

# 5. Understanding Coefficients

Suppose:

$$
Salary=25000+5000(Experience)
$$

The coefficient is:

```text
5000
```

This means that, according to the fitted model, one additional unit of experience is associated with an increase of approximately `₹5,000` in predicted salary, assuming other included variables remain fixed.

### Important Interview Point

A coefficient shows a model relationship. It does **not automatically prove causation**.

---

# 6. Important Hyperparameters

Basic `LinearRegression` has relatively few hyperparameters compared with algorithms such as Random Forest or XGBoost.

| Parameter       | Meaning                               | Effect                                                     |
| --------------- | ------------------------------------- | ---------------------------------------------------------- |
| `fit_intercept` | Whether to calculate intercept        | Usually `True`                                             |
| `copy_X`        | Whether to copy input data            | Mainly affects memory                                      |
| `positive`      | Forces coefficients to be positive    | Useful when domain knowledge requires non-negative effects |
| `n_jobs`        | Number of CPU jobs in supported cases | Can affect computation speed                               |

Example:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(
    fit_intercept=True
)
```

### Important

Linear Regression itself does not have a learning-rate or number-of-trees parameter.

For controlling coefficient magnitude, use regularized models such as:

* Ridge
* Lasso
* ElasticNet

---

# 7. What Happens During `model.fit()`?

When we execute:

```python
model.fit(X_train, y_train)
```

the model:

```text
X_train + y_train
       ↓
Find coefficients
       ↓
Calculate predictions
       ↓
Measure residuals
       ↓
Find coefficients minimizing squared error
       ↓
Store learned coefficients
```

After training:

```python
model.coef_
```

contains the learned feature coefficients.

```python
model.intercept_
```

contains the intercept.

---

# 8. Python Implementation

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data.csv")

# Example:
# Features
X = df[["area", "bedrooms"]]

# Target
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

---

# 9. Evaluation Metrics

## MAE

$$
MAE=\frac{1}{n}\sum |y-\hat{y}|
$$

Measures average absolute prediction error.

Example:

```text
MAE = 5
```

means predictions are off by about 5 target units on average.

---

## MSE

$$
MSE=\frac{1}{n}\sum(y-\hat{y})^2
$$

Large errors receive greater penalty.

---

## RMSE

$$
RMSE=\sqrt{MSE}
$$

RMSE is easier to interpret than MSE because it uses the same units as the target.

---

## R²

$$
R^2=
1-\frac{SS_{res}}{SS_{tot}}
$$

R² compares the model against a baseline that predicts the mean target.

| R²    | General interpretation                          |
| ----- | ----------------------------------------------- |
| `1.0` | Perfect fit                                     |
| `0.8` | Strong explanatory performance in many contexts |
| `0.5` | Moderate                                        |
| `0`   | No improvement over mean baseline               |
| `< 0` | Worse than mean baseline                        |

Do **not** say:

> "R² = 0.8 means the model is 80% accurate."

That is incorrect.

---

# 10. Residual Analysis

Residual:

$$
e_i=y_i-\hat{y_i}
$$

A useful residual plot should ideally show errors scattered around zero without a strong systematic pattern.

```text
Residual
   │  ●      ●
   │      ●
 0 ├──●────────●──────
   │    ●       ●
   │ ●
   └────────────────── Prediction
```

### Warning signs

```text
Curved pattern
→ Possible non-linearity

Increasing spread
→ Possible heteroscedasticity

Extreme points
→ Possible influential observations/outliers
```

Residual analysis is important because a good R² alone does not guarantee that the model is appropriate.

---

# 11. Important Assumptions

## 1. Linearity

The expected target relationship should be reasonably represented by the chosen linear features.

```text
Linear pattern → Good fit possible
Strong curve   → Linear model may underfit
```

---

## 2. Independence

Observations/errors should be appropriately independent for the intended modeling or statistical inference.

This is especially important with:

* Time-series data
* Repeated measurements
* Grouped observations

---

## 3. Homoscedasticity

Residual variance should be reasonably constant.

```text
Good:

●  ● ●  ●
 ● ●  ● ●
────────────
●  ● ●  ●

Problem:

●
 ● ●
  ● ● ●
    ● ● ● ●
```

The second pattern indicates increasing error variance.

---

## 4. Low Multicollinearity

Features should not contain severe redundant linear information.

Example:

```text
Area in sqft
       ↕
Area in square meters
```

These features contain almost the same information.

---

## 5. Normality of Residuals

For prediction alone, normally distributed residuals are not a strict requirement.

Normality becomes more important when performing classical statistical inference such as confidence intervals and hypothesis tests.

---

# 12. Multicollinearity

Multicollinearity occurs when predictor variables are strongly related to each other.

Example:

```text
X1 = Area in sqft
X2 = Area in square meters
```

### Problems

* Coefficients become unstable.
* Coefficient interpretation becomes difficult.
* Small data changes can significantly change coefficients.

### Detection

A common diagnostic is **Variance Inflation Factor (VIF)**.

### Solutions

* Remove redundant features.
* Combine related features.
* Use domain knowledge.
* Use Ridge Regression.
* Use dimensionality reduction when appropriate.

---

# 13. Outliers and Influential Points

Linear Regression can be sensitive to extreme observations because squared errors heavily penalize large residuals.

Example:

```text
● ● ● ● ● ●
● ● ● ●
                         ●
```

That extreme point can influence the fitted line.

### Do not automatically delete outliers.

First determine whether the observation is:

* A data-entry error
* A genuine rare case
* A measurement problem
* An important business case

---

# 14. Feature Scaling

Basic OLS Linear Regression does **not require feature scaling**.

For example:

```text
Age       → 20–60
Income    → 20,000–200,000
```

The model can still train.

However, scaling can be useful when:

* comparing coefficient magnitudes,
* using regularized linear models,
* combining Linear Regression with other algorithms in a pipeline.

---

# 15. Categorical Features

Linear Regression cannot directly work with raw text categories.

Example:

```text
City
----
Pune
Mumbai
Nagpur
```

Encode them first.

A common approach:

```python
from sklearn.preprocessing import OneHotEncoder
```

For mixed numerical and categorical data, a `ColumnTransformer` + `Pipeline` is usually a cleaner production approach.

---

# 16. Overfitting and Underfitting

### Underfitting

```text
Training performance → Low
Testing performance  → Low
```

Possible causes:

* Model too simple
* Missing important features
* Strong nonlinear relationship

---

### Overfitting

```text
Training performance → Very high
Testing performance  → Much lower
```

Possible causes:

* Too many features
* Data leakage
* Noise
* High-dimensional feature engineering

Possible solutions:

* Better feature selection
* Regularization
* Cross-validation
* More representative data

---

# 17. Regularization

Ordinary Linear Regression minimizes prediction error.

Regularization adds a penalty to discourage unnecessarily large coefficients.

### Ridge

$$
Loss = RSS+\lambda\sum\beta_j^2
$$

Ridge uses **L2 regularization**.

### Lasso

$$
Loss = RSS+\lambda\sum|\beta_j|
$$

Lasso uses **L1 regularization**.

### Difference

```text
Linear Regression
       ↓
No regularization penalty

Ridge
       ↓
L2 penalty
       ↓
Shrinks coefficients

Lasso
       ↓
L1 penalty
       ↓
Can make some coefficients exactly zero
```

---

# 18. Cross-Validation

A single train-test split can give a misleading estimate.

Use cross-validation to evaluate performance across multiple splits.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Scores:", scores)
print("Mean R²:", scores.mean())
```

### Why?

```text
Dataset
   ↓
Fold 1 → Train / Validate
Fold 2 → Train / Validate
Fold 3 → Train / Validate
Fold 4 → Train / Validate
Fold 5 → Train / Validate
   ↓
Average performance
```

This gives a more stable estimate of model performance.

---

# 19. Data Leakage

Never allow test information to influence model training.

Incorrect:

```text
Complete Dataset
      ↓
Scaling / Imputation
      ↓
Train-Test Split
```

Better:

```text
Complete Dataset
      ↓
Train-Test Split
      ↓
Fit preprocessing on Train
      ↓
Transform Train + Test
      ↓
Train Model
      ↓
Evaluate Test
```

For production workflows, use a Pipeline.

---

# 20. Advantages

1. Simple to understand.
2. Fast to train.
3. Fast to predict.
4. Coefficients are interpretable.
5. Good baseline model.
6. Works well when relationships are approximately linear.
7. Easy to combine with preprocessing pipelines.

---

# 21. Disadvantages

1. Limited when relationships are strongly nonlinear.
2. Sensitive to influential outliers.
3. Multicollinearity can make coefficients unstable.
4. Requires appropriate feature representation.
5. Can underfit complex datasets.
6. Coefficient interpretation can become difficult with correlated features.
7. Good training performance does not guarantee good generalization.

---

# 22. When NOT to Use Linear Regression

Avoid using basic Linear Regression as the main model when:

* The relationship is strongly nonlinear.
* There are complex feature interactions.
* The dataset contains many influential outliers.
* The problem requires a nonlinear decision boundary.
* Another model clearly provides better validated performance.

Possible alternatives:

```text
Nonlinear relationship
        ↓
Polynomial Regression
        ↓
Random Forest
        ↓
Gradient Boosting
        ↓
XGBoost
```

The choice should be based on validation performance, data characteristics, interpretability requirements, and computational constraints.

---

# 23. Linear Regression vs Related Algorithms

| Feature                    | Linear Regression    | Ridge               | Lasso                      | Random Forest                |
| -------------------------- | -------------------- | ------------------- | -------------------------- | ---------------------------- |
| Regularization             | No                   | L2                  | L1                         | Built-in ensemble behavior   |
| Nonlinear patterns         | Poor                 | Poor                | Poor                       | Strong                       |
| Coefficient interpretation | High                 | High                | High                       | Low                          |
| Scaling required           | No                   | Usually useful      | Usually useful             | No                           |
| Multicollinearity          | Sensitive            | Handles better      | Can help                   | Generally less sensitive     |
| Feature selection          | No                   | No direct selection | Yes, can zero coefficients | Feature importance available |
| Training speed             | Very fast            | Very fast           | Fast                       | Usually slower               |
| Best use                   | Linear relationships | Correlated features | Sparse models              | Complex nonlinear data       |

---

# 24. Real-World Use Cases

* House price prediction
* Sales forecasting
* Demand estimation
* Salary prediction
* Revenue prediction
* Cost estimation
* Energy consumption prediction
* Business trend analysis

---

# 25. Common Mistakes

### Mistake 1 — Calling R² "accuracy"

Wrong:

```text
R² = 75%
→ Accuracy = 75%
```

R² is not classification accuracy.

---

### Mistake 2 — Ignoring residuals

A high R² does not guarantee a suitable linear relationship.

---

### Mistake 3 — Training before splitting

This can cause data leakage.

---

### Mistake 4 — Blindly removing outliers

Investigate them first.

---

### Mistake 5 — Interpreting coefficients without considering units

A coefficient's magnitude depends on the feature's scale.

---

### Mistake 6 — Assuming correlation means causation

A regression coefficient does not automatically establish a causal relationship.

---

### Mistake 7 — Using Linear Regression for every regression problem

Always compare against suitable alternatives.

---

# 26. Practical Experiment

Use a house-price dataset.

```text
Dataset
   ↓
Clean missing values
   ↓
Select numerical features
   ↓
Train/Test Split
   ↓
Linear Regression
   ↓
MAE + RMSE + R²
   ↓
Residual Analysis
   ↓
Cross-Validation
   ↓
Compare with Ridge / Random Forest / XGBoost
```

### Experiment 1 — Feature comparison

Train with:

```text
Area
```

Then:

```text
Area + BHK
```

Then:

```text
Area + BHK + Bathrooms
```

Compare R², MAE, and RMSE.

---

### Experiment 2 — Outlier effect

Train:

```text
Original dataset
```

Then compare with:

```text
Dataset after justified outlier treatment
```

Observe how coefficients and metrics change.

---

### Experiment 3 — Regularization

Compare:

```text
Linear Regression
vs
Ridge
vs
Lasso
```

Observe coefficient changes.

---

### Experiment 4 — Model comparison

Compare:

```text
Linear Regression
Random Forest
XGBoost
```

Use the same train/test strategy and evaluation metrics.

---

# 27. Interview Questions

## Basic

### 1. What is Linear Regression?

**Answer:** Linear Regression is a supervised learning algorithm used to predict a continuous numerical target by modeling it as a linear combination of input features.

### 2. Is Linear Regression supervised or unsupervised?

**Answer:** Supervised, because it learns from labeled input-output pairs.

### 3. Is Linear Regression classification or regression?

**Answer:** Regression.

### 4. What does Linear Regression predict?

**Answer:** A continuous numerical value such as price, salary, sales, or demand.

### 5. What is a residual?

**Answer:** The difference between the actual value and predicted value.

$$
Residual=y-\hat{y}
$$

---

## Intermediate

### 6. What does OLS minimize?

**Answer:** The sum of squared residuals.

### 7. What is the meaning of a coefficient?

**Answer:** It represents the expected change in the predicted target for a one-unit increase in that feature, while holding the other included features constant.

### 8. Does Linear Regression require feature scaling?

**Answer:** Basic OLS Linear Regression does not require scaling.

### 9. What is multicollinearity?

**Answer:** It occurs when predictor variables are strongly linearly related, making coefficient estimates unstable or difficult to interpret.

### 10. Why can outliers affect Linear Regression?

**Answer:** Because the squared-error objective gives large residuals disproportionately high influence.

---

## Technical

### 11. What is the difference between MAE and RMSE?

**Answer:** Both measure prediction error, but RMSE penalizes large errors more strongly because errors are squared before averaging.

### 12. Can R² be negative?

**Answer:** Yes. On evaluation data, R² can be negative when the model performs worse than the mean-prediction baseline.

### 13. What is the Normal Equation?

**Answer:** It is a closed-form expression for obtaining OLS coefficients:

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

### 14. What is the difference between Linear Regression and Ridge?

**Answer:** Ridge adds an L2 penalty to the loss function, which discourages large coefficients and can help with multicollinearity.

### 15. Why use cross-validation?

**Answer:** It evaluates the model across multiple train-validation splits and provides a more stable estimate of generalization performance.

---

# 28. Common Interview Traps

### Trap 1

> "R² = 0.8 means 80% accuracy."

**Wrong.**

R² measures performance relative to a mean-prediction baseline. It is not classification accuracy.

---

### Trap 2

> "Linear Regression always needs feature scaling."

**Wrong.**

Basic OLS Linear Regression does not require scaling.

---

### Trap 3

> "Linear Regression only works with one feature."

**Wrong.**

It can use many features. That is Multiple Linear Regression.

---

### Trap 4

> "Polynomial Regression is completely different from Linear Regression."

**Not exactly.**

Polynomial features introduce powers such as `x²` and `x³`, but the resulting model is still linear in its coefficients.

---

### Trap 5

> "High R² means the model is always good."

**Wrong.**

You should also consider:

* Test performance
* Cross-validation
* Residuals
* Data leakage
* Outliers
* Business context
* Generalization

---

# 29. One-Minute Interview Explanation

> "Linear Regression is a supervised learning algorithm used for predicting continuous values. It assumes that the target can be represented as a linear combination of the input features. During training, the model learns coefficients and an intercept. In Ordinary Least Squares, these parameters are selected to minimize the sum of squared residuals between actual and predicted values. After training, we use the learned coefficients to make predictions on unseen data. I would evaluate a regression model using metrics such as MAE, RMSE, and R², and I would also check residuals and cross-validation performance. If there is strong multicollinearity or overfitting, I would consider regularized models such as Ridge or Lasso."

---

# 30. Quick Revision

```text
Algorithm:
Linear Regression

Type:
Supervised Learning

Problem:
Regression

Target:
Continuous numerical value

Main idea:
Learn the best linear relationship between features and target.

Formula:
ŷ = b₀ + b₁x₁ + ... + bₙxₙ

Optimization:
Ordinary Least Squares

Main objective:
Minimize squared residuals

Important metrics:
MAE
MSE
RMSE
R²

Scaling:
Not required for basic OLS

Important concepts:
Residuals
Multicollinearity
Outliers
Homoscedasticity
Overfitting
Underfitting
Regularization
Cross-validation

Main advantage:
Simple and interpretable

Main limitation:
Poor at complex nonlinear relationships

Related algorithms:
Ridge
Lasso
ElasticNet
Polynomial Regression
Random Forest
XGBoost
```

---

# 31. Final Revision Flow

```text
             Linear Regression
                     ↓
          Supervised Learning
                     ↓
              Regression
                     ↓
        Continuous Target Value
                     ↓
       Linear Relationship Assumption
                     ↓
            Learn Coefficients
                     ↓
          Ordinary Least Squares
                     ↓
        Minimize Squared Residuals
                     ↓
                Prediction
                     ↓
          MAE / RMSE / R²
                     ↓
      Residual + Generalization Check
                     ↓
       Ridge / Lasso if Appropriate
                     ↓
              Interview Ready
```

---

## Repository Structure

```text
01_Linear_Regression/
│
├── README.md
└── linear_regression.ipynb
```

**Recommended next topic:**

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
```

