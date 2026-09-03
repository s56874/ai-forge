# Lasso Regression

## 1. What is Lasso Regression?

**Lasso Regression** is a regularized version of Linear Regression.

Lasso stands for:

> **Least Absolute Shrinkage and Selection Operator**

It adds an **L1 regularization penalty** to the Linear Regression cost function.

The main purpose of Lasso Regression is:

* Reduce overfitting
* Control large coefficients
* Perform **feature selection**
* Handle datasets with many features

---

## 2. Basic Idea

Normal Linear Regression tries to find coefficients that minimize prediction error.

Lasso adds an additional penalty for large coefficients.

### Linear Regression

```text
Minimize → Prediction Error
```

### Lasso Regression

```text
Minimize → Prediction Error + L1 Penalty
```

This encourages some coefficients to become exactly **0**.

---

## 3. Mathematical Formula

Linear Regression:

```text
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Lasso objective:

```text
RSS + α Σ|bⱼ|
```

Where:

* `RSS` = Residual Sum of Squares
* `α` = regularization strength
* `bⱼ` = model coefficients
* `|bⱼ|` = absolute value of coefficient

---

## 4. What is α?

`alpha` controls the strength of regularization.

### Small α

```text
Less regularization
↓
Coefficients remain larger
↓
Higher risk of overfitting
```

### Large α

```text
More regularization
↓
Coefficients become smaller
↓
Some coefficients may become 0
↓
Higher risk of underfitting
```

Example:

```python
Lasso(alpha=0.1)
```

---

## 5. Why Does Lasso Set Coefficients to Zero?

This is the most important feature of Lasso.

Suppose the model has:

```text
Area       → 0.85
Bedrooms   → 0.32
Balcony    → 0.00
Parking    → 0.00
```

Lasso has effectively removed `Balcony` and `Parking` from the model.

Therefore:

> **Lasso can automatically perform feature selection.**

---

## 6. Lasso vs Linear Regression

| Linear Regression              | Lasso Regression           |
| ------------------------------ | -------------------------- |
| No regularization              | L1 regularization          |
| Can overfit                    | Helps reduce overfitting   |
| Keeps coefficients             | Can make coefficients zero |
| No automatic feature selection | Performs feature selection |
| Simple baseline                | Useful with many features  |

---

## 7. Python Implementation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error

X = df[["area", "bhk", "bathroom", "balcony"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Lasso(alpha=0.1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
```

---

## 8. Check Coefficients

You can see which features Lasso selected:

```python
print(model.coef_)
```

For feature names:

```python
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, coefficient)
```

Example:

```text
area       0.52
bhk        1.21
bathroom   0.00
balcony    0.00
```

Here, Lasso has assigned zero coefficient to `bathroom` and `balcony`.

---

## 9. Feature Scaling

Feature scaling is **important for Lasso**.

Why?

Because the regularization penalty is applied to the coefficients.

If features have very different scales, the penalty can affect them unfairly.

Use `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso

model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 10. Best Graph

For regression, one of the most useful graphs is:

### Actual vs Predicted

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression: Actual vs Predicted")

plt.show()
```

### Interpretation

If predictions are good:

```text
Points → close to diagonal line
```

If predictions are poor:

```text
Points → far from diagonal line
```

---

## 11. Residual Plot

Residual:

```text
Residual = Actual - Predicted
```

Simple residual plot:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=y_pred, y=y_test - y_pred)

plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Lasso Regression: Residual Plot")

plt.show()
```

Ideally, residuals should be randomly distributed around zero.

---

## 12. Important Hyperparameter

The main Lasso hyperparameter is:

```python
alpha
```

Example:

```python
Lasso(alpha=0.01)
Lasso(alpha=0.1)
Lasso(alpha=1)
Lasso(alpha=10)
```

You can test different values and choose the one that performs best on validation data.

---

## 13. Lasso and Multicollinearity

Lasso can be useful when many features are correlated.

For example:

```text
Area
Total Sqft
Rooms
Bedrooms
```

Some of these may contain overlapping information.

Lasso can shrink some coefficients toward zero.

However, when many features are strongly correlated, which feature gets selected can be unstable.

---

## 14. Lasso vs Ridge

This is a very common interview question.

| Lasso                           | Ridge                                |   |           |
| ------------------------------- | ------------------------------------ | - | --------- |
| L1 regularization               | L2 regularization                    |   |           |
| Uses `                          | β                                    | ` | Uses `β²` |
| Can make coefficients exactly 0 | Usually does not make them exactly 0 |   |           |
| Performs feature selection      | Keeps all features                   |   |           |
| Useful for sparse models        | Useful with correlated features      |   |           |

### Remember

```text
Lasso → L1 → Zero coefficients → Feature Selection

Ridge → L2 → Smaller coefficients → No exact zero usually
```

---

## 15. Lasso vs Elastic Net

Elastic Net combines L1 and L2 regularization.

```text
Lasso       → L1
Ridge       → L2
Elastic Net → L1 + L2
```

Elastic Net can be useful when you want both:

* Feature selection
* Better handling of correlated features

---

## 16. Advantages

* Reduces overfitting
* Performs feature selection
* Produces simpler models
* Useful when there are many features
* Easy to implement with Scikit-learn
* Improves interpretability by removing some features

---

## 17. Disadvantages

* Sensitive to feature scaling
* Large `alpha` can cause underfitting
* Feature selection can be unstable with highly correlated features
* Not ideal for strongly nonlinear relationships without feature engineering
* Requires tuning of `alpha`

---

## 18. When to Use Lasso

Use Lasso when:

* You have many features
* Some features may be irrelevant
* You want automatic feature selection
* You want a simpler model
* You want to reduce overfitting

---

## 19. When Not to Use Lasso

Avoid relying on Lasso alone when:

* The relationship is highly nonlinear
* You have strong complex interactions
* You have many correlated features and want to keep groups of them
* A tree-based model performs much better for your problem

---

## 20. Important Interview Questions

### Q1. What is Lasso Regression?

**Answer:**

> Lasso Regression is a regularized Linear Regression algorithm that uses L1 regularization. It reduces overfitting and can make some feature coefficients exactly zero, which allows automatic feature selection.

---

### Q2. What does L1 regularization mean?

**Answer:**

> L1 regularization adds the sum of the absolute values of the coefficients to the loss function.

---

### Q3. Why is Lasso useful for feature selection?

**Answer:**

> Lasso can shrink some coefficients exactly to zero. Features with zero coefficients effectively do not contribute to the model prediction.

---

### Q4. What happens when alpha increases?

**Answer:**

> Regularization becomes stronger, coefficients shrink more, and more coefficients may become zero. If alpha is too large, the model can underfit.

---

### Q5. Does Lasso require feature scaling?

**Answer:**

> Yes, feature scaling is generally recommended because Lasso penalizes coefficients and different feature scales can affect the regularization unfairly.

---

### Q6. Is Lasso supervised or unsupervised?

**Answer:**

> Lasso Regression is a **supervised learning** algorithm because it learns from input features and a known target variable.

---

### Q7. Is Lasso classification or regression?

**Answer:**

> Lasso Regression is primarily a **regression algorithm**. L1 regularization is also used in classification models such as Logistic Regression.

---

## 21. Common Interview Trap

❌ Wrong:

> Lasso removes columns from the original dataset automatically.

✅ Better:

> Lasso assigns some feature coefficients to zero. This effectively removes their contribution to the model, although the original dataset columns are not physically deleted.

---

## 22. Lasso in a House Price Project

For house price prediction, Lasso can be used when there are multiple features such as:

```text
Area
BHK
Bathroom
Balcony
Location
```

A good project explanation:

> "I used Lasso Regression as a regularized linear model. It helped reduce overfitting and identify less useful features by shrinking some coefficients toward zero. I compared its performance with other regression models using R², MAE, and RMSE."

---

## 23. Quick Revision

```text
Lasso Regression
       ↓
Linear Regression + L1 Regularization
       ↓
Controls large coefficients
       ↓
Some coefficients can become 0
       ↓
Feature Selection
       ↓
Reduces Overfitting
```

### Remember This

> **Lasso = L1 = coefficients can become zero = feature selection.**
