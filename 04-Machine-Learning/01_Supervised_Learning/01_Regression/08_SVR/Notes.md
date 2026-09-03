# Support Vector Regression (SVR)

## 1. What is SVR?

**SVR (Support Vector Regression)** is the regression version of **Support Vector Machine (SVM)**.

It is used to predict a **continuous numerical value**.

Examples:

* House price prediction
* Salary prediction
* Stock price prediction
* Temperature prediction

SVR tries to find a function that predicts values while allowing a small amount of error within an **ε (epsilon) margin**.

---

## 2. Basic Idea

Instead of trying to minimize every small error, SVR creates an **ε-tube** around the prediction function.

```text
        ε-tube
   ─────────────────
          Prediction
   ─────────────────
        ε-tube
```

Predictions inside this tube are considered acceptable.

Points outside the tube contribute to the optimization.

---

## 3. Mathematical Formula

For linear SVR:

```text
ŷ = wᵀx + b
```

SVR tries to find a function:

```text
f(x) = wᵀx + b
```

while keeping prediction errors within the ε margin.

The optimization includes:

```text
1/2 ||w||²
```

plus penalties for errors outside the ε-tube.

---

## 4. What is ε?

`epsilon` controls the width of the error tolerance.

```python
SVR(epsilon=0.1)
```

### Small epsilon

```text
Small tolerance
↓
More points may become support vectors
↓
More detailed model
```

### Large epsilon

```text
Larger tolerance
↓
Fewer points may influence the model
↓
Simpler model
```

---

## 5. What are Support Vectors?

**Support vectors** are training data points that are important for defining the regression function.

Points well inside the ε-tube generally do not affect the solution in the same way.

The points near or outside the ε-tube are especially important.

```text
Support vectors
      ↓
Important boundary/error points
      ↓
Help determine the regression function
```

---

## 6. Important SVR Parameters

### `C`

Controls the penalty for errors outside the ε-tube.

```python
SVR(C=10)
```

Higher `C`:

```text
More penalty for errors
↓
Model tries harder to fit training data
↓
Can increase overfitting
```

Lower `C`:

```text
More tolerance for errors
↓
Simpler model
↓
Can increase underfitting
```

---

### `epsilon`

Controls the width of the ε-tube.

```python
SVR(epsilon=0.1)
```

---

### `kernel`

Controls the type of transformation used by SVR.

Common choices:

```text
linear
rbf
poly
sigmoid
```

The most commonly useful choice for nonlinear problems is:

```python
kernel="rbf"
```

---

### `gamma`

Important for the RBF kernel.

```python
SVR(kernel="rbf", gamma="scale")
```

It controls how strongly individual training points influence the model.

Higher `gamma`:

```text
Smaller influence area
↓
More complex model
↓
Higher overfitting risk
```

Lower `gamma`:

```text
Larger influence area
↓
Smoother model
```

---

## 7. Why Feature Scaling is Important

Feature scaling is **very important for SVR**.

SVR uses distances between data points, especially with kernels such as RBF.

If one feature has a much larger scale than another, it can dominate the model.

Use `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
```

---

## 8. Simple SVR Example

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

X = df[["area"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
```

---

## 9. Best Graph

For regression, use **Actual vs Predicted**.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("SVR: Actual vs Predicted")

plt.show()
```

### Interpretation

```text
Points close to diagonal
        ↓
Better predictions
```

---

## 10. Residual Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=y_pred, y=y_test - y_pred)

plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("SVR: Residual Plot")

plt.show()
```

Residual:

```text
Residual = Actual - Predicted
```

---

## 11. SVR Kernels

### Linear Kernel

```python
SVR(kernel="linear")
```

Useful when the relationship is approximately linear.

### RBF Kernel

```python
SVR(kernel="rbf")
```

Useful for nonlinear relationships.

### Polynomial Kernel

```python
SVR(kernel="poly")
```

Can model polynomial relationships.

---

## 12. Complete Tuned SVR Model

Instead of manually selecting `C`, `epsilon`, and `gamma`, use **GridSearchCV**.

### Step 1: Imports

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
```

### Step 2: Features and Target

```python
X = df[["area", "bhk", "bathroom", "balcony"]]
y = df["price"]
```

### Step 3: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Step 4: Create Pipeline

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])
```

### Step 5: Define Parameters

```python
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma": ["scale", 0.01, 0.1, 1]
}
```

### Step 6: GridSearchCV

```python
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

### Step 7: Get Best Model

```python
model = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)

print("Best CV R²:")
print(grid_search.best_score_)
```

Example:

```text
Best Parameters:
{
    'svr__C': 10,
    'svr__epsilon': 0.1,
    'svr__gamma': 'scale'
}
```

The exact values depend on the dataset.

### Step 8: Prediction

```python
y_pred = model.predict(X_test)
```

### Step 9: Final Evaluation

```python
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Testing R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
```

---

## 13. Complete Tuned SVR Code

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import matplotlib.pyplot as plt

# Features and target
X = df[["area", "bhk", "bathroom", "balcony"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])

# Hyperparameters
param_grid = {
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma": ["scale", 0.01, 0.1, 1]
}

# Hyperparameter tuning
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Best Parameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)
print("Testing R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# Actual vs Predicted
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Tuned SVR: Actual vs Predicted")

plt.show()
```

---

## 14. SVR Overfitting and Underfitting

### Overfitting

Can happen with:

```text
High C
High gamma
Very small epsilon
```

### Underfitting

Can happen with:

```text
Very low C
Very low gamma
Very large epsilon
```

These are general tendencies; the best combination depends on the dataset.

---

## 15. SVR vs Linear Regression

| Linear Regression            | SVR                               |
| ---------------------------- | --------------------------------- |
| Linear relationship          | Can model nonlinear relationships |
| Simple and fast              | More computationally expensive    |
| No kernel                    | Uses kernels                      |
| Scaling usually not critical | Scaling is important              |
| Easier to interpret          | Less interpretable                |

---

## 16. SVR vs Random Forest

| SVR                                | Random Forest                               |
| ---------------------------------- | ------------------------------------------- |
| Kernel-based                       | Tree-based                                  |
| Scaling important                  | Scaling usually unnecessary                 |
| Can model nonlinear patterns       | Can model nonlinear patterns                |
| Sensitive to hyperparameters       | Also requires tuning                        |
| Can be expensive on large datasets | Usually handles large tabular datasets well |
| Uses support vectors               | Uses many decision trees                    |

---

## 17. Advantages

* Can model nonlinear relationships
* Effective in high-dimensional feature spaces
* Uses kernels for complex relationships
* Works well when the dataset is not extremely large
* Has good control over model complexity

---

## 18. Disadvantages

* Feature scaling is important
* Hyperparameter tuning can be expensive
* Training can become slow on large datasets
* Less interpretable than Linear Regression
* Sensitive to `C`, `epsilon`, and `gamma`

---

## 19. When to Use SVR

Use SVR when:

* The dataset is small or medium-sized
* The relationship is nonlinear
* You want a powerful regression model
* Kernel methods are suitable for the problem
* You are willing to tune hyperparameters

---

## 20. When Not to Use SVR

Avoid SVR when:

* The dataset is extremely large
* Training speed is very important
* You need highly interpretable coefficients
* A simpler model already performs sufficiently well

---

## 21. Important Interview Questions

### Q1. What is SVR?

> SVR, or Support Vector Regression, is the regression version of Support Vector Machine. It tries to find a function where most predictions fall within an epsilon-insensitive tube while controlling model complexity.

### Q2. What is the ε-tube?

> The ε-tube defines a region around the prediction function where errors are ignored up to the specified epsilon value.

### Q3. What are support vectors?

> Support vectors are important training points near or outside the epsilon tube that influence the final regression function.

### Q4. Why is scaling important in SVR?

> SVR, especially kernel-based SVR, depends on distances between observations. Features with different scales can therefore distort the model, so standardization is generally recommended.

### Q5. What does C control?

> `C` controls the penalty for errors outside the epsilon tube. A larger C generally makes the model try harder to fit the training data.

### Q6. What does gamma do?

> For kernels such as RBF, gamma controls how strongly individual training points influence the model. Higher gamma generally produces a more localized and complex decision function.

### Q7. Why use RBF kernel?

> RBF can model nonlinear relationships without explicitly creating polynomial features manually.

### Q8. How do you tune SVR?

> I use cross-validation with GridSearchCV or RandomizedSearchCV to tune parameters such as `C`, `epsilon`, and `gamma`.

---

## 22. Common Interview Trap

❌ Wrong:

> SVR tries to make every prediction exactly equal to the actual value.

✅ Better:

> SVR allows errors within an epsilon margin and focuses on controlling errors outside that margin while maintaining a suitable model complexity.

---

## 23. House Price Project Explanation

> "I used SVR as another regression model for house price prediction. Since house prices can have nonlinear relationships with features such as area, BHK, and bathrooms, I tested an RBF kernel. I standardized the features and tuned C, epsilon, and gamma using GridSearchCV. I evaluated the final model using R², MAE, and RMSE."

---

## 24. Quick Revision

```text
SVR
 ↓
Support Vector Regression
 ↓
Regression version of SVM
 ↓
ε-tube
 ↓
Support Vectors
 ↓
Kernel → nonlinear relationships
 ↓
Important parameters:
C, epsilon, gamma
 ↓
Scaling is important
```

### Remember This

> **SVR = SVM for Regression + ε-tube + Support Vectors + Kernels**
