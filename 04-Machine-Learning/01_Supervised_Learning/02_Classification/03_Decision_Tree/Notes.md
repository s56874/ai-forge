# Logistic Regression — Complete Notes

## 1. What is Logistic Regression?

**Logistic Regression** is a **supervised machine learning algorithm** mainly used for **classification problems**.

It predicts the probability that an observation belongs to a particular class.

For our Loan Approval problem:

```text
0 → Loan Not Approved
1 → Loan Approved
```

The model calculates the probability of loan approval and then converts that probability into a class.

### Example

```text
Probability = 0.82
```

The model predicts:

```text
Loan Approved
```

### Important Point

Despite its name, **Logistic Regression is mainly used for classification**, not ordinary regression.

---

# 2. Basic Idea

The basic idea is:

```text
Input Features
      ↓
Linear Equation
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Classification
```

For example, the model receives:

```text
Age
Salary
Years Experience
Credit Score
Loan Amount
```

It combines these features using learned coefficients.

Then it applies the **sigmoid function** to convert the result into a value between:

```text
0 and 1
```

Example:

```text
0.85 → High probability of approval
0.20 → Low probability of approval
```

---

# 3. Mathematical Formula

Logistic Regression first calculates a linear score:

```text
z = β0 + β1X1 + β2X2 + ... + βnXn
```

Where:

```text
β0 → Intercept
β1, β2, ... → Coefficients
X1, X2, ... → Input Features
```

For our Loan Approval dataset:

```text
z =
β0
+ β1(age)
+ β2(salary)
+ β3(years_experience)
+ β4(credit_score)
+ β5(loan_amount)
```

This `z` is then passed through the sigmoid function.

```text
z
↓
Sigmoid
↓
Probability
```

---

# 4. What is Sigmoid?

The **sigmoid function** converts any real-valued number into a value between **0 and 1**.

Formula:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

### Examples

If:

```text
z = very large positive number
```

then:

```text
Sigmoid(z) ≈ 1
```

If:

```text
z = very large negative number
```

then:

```text
Sigmoid(z) ≈ 0
```

If:

```text
z = 0
```

then:

```text
Sigmoid(0) = 0.5
```

### Simple Understanding

```text
Linear Score
     ↓
 Sigmoid
     ↓
0 to 1 probability
```

---

# 5. What is Classification Threshold?

After getting a probability, we need to convert it into a class.

A common threshold is:

```text
0.5
```

For binary classification:

```text
Probability >= 0.5
        ↓
Class 1
```

and:

```text
Probability < 0.5
        ↓
Class 0
```

For Loan Approval:

```text
Probability = 0.80
→ Loan Approved

Probability = 0.30
→ Loan Not Approved
```

### Important

The threshold does not have to be 0.5 in every application. It can be changed depending on the problem and the cost of false positives/false negatives.

---

# 6. What are Coefficients?

Logistic Regression learns a coefficient for each feature.

For example:

```text
age               → β1
salary            → β2
years_experience  → β3
credit_score      → β4
loan_amount       → β5
```

A coefficient tells us how a feature affects the **log-odds** of the positive class, holding other features constant.

### Positive coefficient

A positive coefficient means increasing that feature increases the model's log-odds for class 1, all else equal.

### Negative coefficient

A negative coefficient means increasing that feature decreases the model's log-odds for class 1, all else equal.

### Important Interview Point

Do not simply say:

> "Positive coefficient means the feature directly increases probability."

The more precise statement is:

> "A positive coefficient increases the log-odds of the positive class, while a negative coefficient decreases them, assuming other features remain constant."

---

# 7. Important Logistic Regression Parameters

Some important parameters of `LogisticRegression` are:

## 1. `C`

Controls the strength of regularization.

```text
Smaller C
→ Stronger regularization

Larger C
→ Weaker regularization
```

---

## 2. `penalty`

Specifies the type of regularization.

Common options include:

```text
l1
l2
```

---

## 3. `solver`

Specifies the optimization algorithm.

Examples:

```text
lbfgs
liblinear
saga
```

The compatible solver depends on the chosen penalty and other settings.

---

## 4. `max_iter`

Maximum number of iterations allowed for optimization.

Example:

```python
max_iter=1000
```

This can help when the model needs more iterations to converge.

---

# 8. Why Feature Scaling is Important

Feature scaling is often useful for Logistic Regression, especially when features have very different numerical ranges.

Our features include:

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
Age → tens
Salary → tens of thousands
Loan Amount → hundreds of thousands
```

We can use:

```python
StandardScaler()
```

Standardization transforms features approximately to:

```text
Mean = 0
Standard Deviation = 1
```

### Why is this useful?

It can:

* Help optimization converge
* Put features on comparable scales
* Make regularization behave more consistently across features

### Interview Answer

> "Feature scaling is useful in Logistic Regression because it can improve optimization and makes regularization behave more consistently when features have different scales."

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
X → Input Features
y → Target
```

So:

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

The dataset has:

```text
No missing values
All features are numerical
No categorical encoding required
```

---

# 10. Simple Logistic Regression Example

The basic workflow is:

```text
Dataset
   ↓
Features + Target
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Probability
   ↓
Class Prediction
```

Conceptually:

```text
Customer Data
     ↓
Linear Score
     ↓
Sigmoid
     ↓
Probability
     ↓
Threshold
     ↓
Loan Approved / Not Approved
```

For example:

```text
Predicted probability = 0.78
```

Using a 0.5 threshold:

```text
0.78 >= 0.5
```

Therefore:

```text
Loan Approved
```

---

# 11. `predict()` vs `predict_proba()`

## `predict()`

Returns the final predicted class.

Example:

```python
y_pred = model.predict(X_test)
```

Output:

```text
0
1
1
0
...
```

---

## `predict_proba()`

Returns probabilities for each class.

Example:

```python
y_prob = model.predict_proba(X_test)
```

A result could look like:

```text
[0.20, 0.80]
```

Meaning:

```text
Class 0 → 20%
Class 1 → 80%
```

For the probability of class 1:

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

### Main Difference

```text
predict()
→ Final class

predict_proba()
→ Probability for each class
```

---

# 12. Best Graph

For Logistic Regression classification, useful visualizations include:

### 1. Confusion Matrix

Shows:

```text
Actual vs Predicted
```

It helps identify:

```text
True Positives
True Negatives
False Positives
False Negatives
```

### 2. ROC Curve

The ROC curve shows the relationship between:

```text
True Positive Rate
```

and:

```text
False Positive Rate
```

at different classification thresholds.

### 3. Probability Distribution

We can also visualize predicted probabilities to understand how confidently the model separates the two classes.

---

# 13. Confusion Matrix

For binary Loan Approval:

```text
                 Predicted
              0          1

Actual 0     TN         FP

Actual 1     FN         TP
```

### TN — True Negative

Actual:

```text
0
```

Predicted:

```text
0
```

Correctly predicted not approved.

### TP — True Positive

Actual:

```text
1
```

Predicted:

```text
1
```

Correctly predicted approved.

### FP — False Positive

Actual:

```text
0
```

Predicted:

```text
1
```

The model predicted approval when the actual label was not approved.

### FN — False Negative

Actual:

```text
1
```

Predicted:

```text
0
```

The model predicted not approved when the actual label was approved.

---

# 14. Classification Evaluation Metrics

## 1. Accuracy

Measures the proportion of all predictions that are correct.

```text
Accuracy = (TP + TN)
           ----------------
           TP + TN + FP + FN
```

---

## 2. Precision

Out of all predicted positive cases, how many were actually positive?

```text
Precision = TP
            ------
            TP + FP
```

---

## 3. Recall

Out of all actual positive cases, how many were correctly identified?

```text
Recall = TP
         ------
         TP + FN
```

---

## 4. F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall
     ------------------------
     Precision + Recall
```

---

## 5. ROC-AUC

ROC-AUC measures how well the model separates the two classes across different probability thresholds.

A higher AUC generally indicates better ranking/separation ability.

---

# 15. Complete Evaluation Code

Use the model's predictions and probabilities to calculate the main metrics:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

Because this dataset has only **30 rows**, evaluation results can vary considerably depending on the train-test split.

---

# 16. Complete Logistic Regression Model

The complete model follows:

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
Logistic Regression
      ↓
Learn Coefficients
      ↓
Calculate Linear Score
      ↓
Sigmoid
      ↓
Probability
      ↓
Threshold
      ↓
Prediction
```

For a clean implementation, we can combine scaling and Logistic Regression using a `Pipeline`.

This also helps prevent data leakage when the preprocessing is used with cross-validation.



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
Logistic Regression
 ↓
Train
 ↓
predict()
 ↓
predict_proba()
 ↓
Evaluation
```

---

# 18. Logistic Regression Overfitting and Underfitting

## Overfitting

Overfitting occurs when the model learns the training data too closely and does not generalize well to unseen data.

Possible signs:

```text
Training Performance → Very High
Test Performance → Much Lower
```

Regularization can help reduce overfitting.

---

## Underfitting

Underfitting occurs when the model is too simple to capture important patterns.

Possible signs:

```text
Training Performance → Low
Test Performance → Low
```

### Important Parameters

Regularization strength can be controlled using:

```text
C
```

Remember:

```text
Smaller C → Stronger regularization
Larger C → Weaker regularization
```

---

# 19. Logistic Regression Regularization

Regularization helps control model complexity.

Two common types are:

```text
L1
L2
```

## L1 Regularization

Can push some coefficients exactly toward zero and can therefore produce sparse models.

Useful when feature selection/sparsity is desirable.

---

## L2 Regularization

Penalizes large coefficients and generally shrinks them toward zero.

It is commonly used as the default regularization approach in many Logistic Regression implementations.

### Why use regularization?

To reduce the risk of:

```text
Overfitting
```

and control excessively large coefficients.

---

# 20. Cross-Validation

Cross-validation helps estimate model performance across multiple training/validation splits.

Example:

```text
5-Fold Cross-Validation
```

The training data is divided into 5 folds.

```text
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
```

Each fold gets a turn as validation data.

Then the scores are averaged.

### Why use it?

It gives a more reliable estimate than depending on only one validation split.

It can also help tune parameters such as:

```text
C
penalty
solver
```

### Important

If using scaling with cross-validation, put scaling inside a `Pipeline` so that each fold learns scaling parameters only from its training portion.

---

# 21. Logistic Regression vs Linear Regression

| Feature      | Logistic Regression                      | Linear Regression  |
| ------------ | ---------------------------------------- | ------------------ |
| Main use     | Classification                           | Regression         |
| Output       | Probability / class                      | Continuous value   |
| Activation   | Sigmoid                                  | None               |
| Output range | Probability 0–1                          | Unbounded          |
| Example      | Loan Approved                            | House Price        |
| Evaluation   | Accuracy, Precision, Recall, F1, ROC-AUC | MAE, MSE, RMSE, R² |

### Example

Logistic Regression:

```text
Loan Approved?
0 or 1
```

Linear Regression:

```text
House Price
₹50,00,000
```

### Easy Interview Answer

> "Logistic Regression is mainly used for classification, while Linear Regression is used to predict continuous numerical values."

---

# 22. Logistic Regression vs Decision Tree

| Feature             | Logistic Regression            | Decision Tree                |
| ------------------- | ------------------------------ | ---------------------------- |
| Main idea           | Mathematical probability model | Rule-based splits            |
| Decision boundary   | Linear in feature space        | Can be non-linear            |
| Feature scaling     | Often useful                   | Usually not required         |
| Interpretability    | Coefficients                   | If-else rules                |
| Probability output  | Yes                            | Yes                          |
| Non-linear patterns | Limited in basic form          | Handles naturally            |
| Overfitting control | Regularization                 | Depth/pruning/other controls |

### Logistic Regression

```text
Features
 ↓
Linear Combination
 ↓
Sigmoid
 ↓
Probability
 ↓
Class
```

### Decision Tree

```text
Feature condition
      ↓
Another condition
      ↓
Prediction
```

---

# 23. Advantages

### 1. Simple

Logistic Regression is relatively simple to understand and implement.

### 2. Fast

It is usually computationally efficient for many classification problems.

### 3. Probability Output

It can provide class probabilities.

### 4. Interpretable

Coefficients can provide useful information about feature relationships with the log-odds.

### 5. Regularization

It supports regularization to control model complexity.

### 6. Strong Baseline

It is often a useful baseline classification algorithm.

---

# 24. Disadvantages

### 1. Basic Model Has a Linear Decision Boundary

It may not capture complex non-linear relationships without feature engineering or transformations.

### 2. Sensitive to Multicollinearity

Highly correlated features can make coefficient estimates less stable or harder to interpret.

### 3. Scaling Can Be Useful

Features with very different scales can make optimization and regularization less convenient.

### 4. Outliers Can Affect the Model

Extreme observations can influence the fitted coefficients.

### 5. Assumptions About the Relationship

The model assumes a linear relationship between features and the **log-odds** of the target.

---

# 25. When to Use Logistic Regression

Logistic Regression is useful when:

* Target is categorical
* Especially binary classification
* You need probabilities
* You want an interpretable baseline
* The relationship is reasonably compatible with a linear decision boundary
* Dataset size is moderate or large

### Examples

```text
Loan Approval
Spam Detection
Customer Churn
Disease Classification
Fraud Detection
```

---

# 26. When Not to Use Logistic Regression

Basic Logistic Regression may not be the best fit when:

* The relationship is highly non-linear
* Complex feature interactions are important
* A linear decision boundary is inadequate
* The problem requires a more flexible model

In such cases, models such as:

```text
Decision Tree
Random Forest
Gradient Boosting
XGBoost
Neural Networks
```

may be considered depending on the problem.

---

# 27. Important Interview Questions

## Q1. What is Logistic Regression?

**Answer:**

> "Logistic Regression is a supervised classification algorithm that estimates the probability of a class using a logistic sigmoid function."

---

## Q2. Why is it called Logistic Regression if it is used for classification?

**Answer:**

> "It is called Logistic Regression because it models the log-odds of the target using a linear combination of features, but it is commonly used for classification."

---

## Q3. What is the sigmoid function?

**Answer:**

> "The sigmoid function converts a real-valued linear score into a value between 0 and 1, which can be interpreted as a probability for binary classification."

---

## Q4. Why do we use a threshold?

**Answer:**

> "The predicted probability needs to be converted into a class. A threshold such as 0.5 is commonly used, although it can be adjusted depending on the application."

---

## Q5. What is the difference between `predict()` and `predict_proba()`?

**Answer:**

> "`predict()` returns the predicted class, while `predict_proba()` returns the estimated probability for each class."

---

## Q6. What does a positive coefficient mean?

**Answer:**

> "A positive coefficient increases the log-odds of the positive class when other features are held constant."

---

## Q7. What is regularization?

**Answer:**

> "Regularization adds a penalty to the objective to control model complexity and reduce the risk of overfitting."

---

## Q8. What is `C`?

**Answer:**

> "`C` controls the inverse strength of regularization. Smaller C means stronger regularization, while larger C means weaker regularization."

---

## Q9. What is the difference between L1 and L2?

**Answer:**

> "L1 regularization can produce sparse coefficients by pushing some coefficients to zero, while L2 regularization shrinks coefficients toward zero."

---

## Q10. Is Logistic Regression supervised or unsupervised?

**Answer:**

> "Logistic Regression is a supervised learning algorithm because it learns from labeled training data."

---

# 28. Common Interview Trap

### Trap 1

**Interviewer: Is Logistic Regression used for regression?**

Answer:

> "Despite its name, Logistic Regression is mainly used for classification."

---

### Trap 2

**Interviewer: What does sigmoid do?**

Wrong:

> "It predicts the class."

Better:

> "It converts the linear score into a value between 0 and 1 that can be interpreted as a probability."

---

### Trap 3

**Interviewer: Does 0.5 always have to be the threshold?**

Answer:

> "No. 0.5 is a common default for binary classification, but the threshold can be changed depending on the application's requirements."

---

### Trap 4

**Interviewer: What does a coefficient directly represent?**

Answer:

> "A coefficient represents the change in log-odds associated with a one-unit increase in the feature, holding other features constant."

---

### Trap 5

**Interviewer: Does Logistic Regression automatically understand categorical text?**

Answer:

> "No. Categorical variables generally need to be encoded into numerical representations before being used by the model."

---

# 29. Loan Approval Project Explanation

If the interviewer asks:

### "Explain your Logistic Regression Loan Approval project."

You can say:

> "I used Logistic Regression to predict whether a loan would be approved or not. My dataset contains features such as age, salary, years of experience, credit score, and loan amount. The target variable is loan approval, where 0 represents not approved and 1 represents approved.
>
> I divided the data into training and testing sets and used StandardScaler to scale the numerical features. Then I trained a Logistic Regression model. The model first calculates a linear combination of the input features and then uses the sigmoid function to convert that score into a probability of loan approval.
>
> Finally, I used the predicted classes and probabilities to evaluate the model using accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix."

### If interviewer asks:

**"Why did you use Logistic Regression?"**

Answer:

> "Because the target is binary classification and Logistic Regression provides both class predictions and probabilities. It also gives interpretable coefficients."

### If interviewer asks:

**"What was the most important concept you learned?"**

Answer:

> "I learned how Logistic Regression converts a linear score into a probability using the sigmoid function and then uses a classification threshold to make a prediction."

---

# 30. Quick Revision

## Logistic Regression in one line

> **Logistic Regression predicts the probability of a class using a linear combination of features followed by the sigmoid function.**

### Complete flow

```text
Features
   ↓
Linear Equation
   ↓
Linear Score (z)
   ↓
Sigmoid
   ↓
Probability
   ↓
Threshold
   ↓
Class
```

### Important Formula

```text
z = β0 + β1X1 + β2X2 + ... + βnXn
```

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

### Important Parameters

```text
C
penalty
solver
max_iter
```

### Important Metrics

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix
```

### Important Concepts

```text
Sigmoid
Probability
Threshold
Coefficients
Regularization
Cross-Validation
Feature Scaling
```

### Most important interview sentence

> **"Logistic Regression is a supervised classification algorithm that calculates a linear score, converts it into a probability using the sigmoid function, and then uses a classification threshold to predict the class."**

---

# Remember This

```text
LOGISTIC REGRESSION

Input Features
      ↓
Linear Combination
      ↓
z = β0 + β1X1 + ... + βnXn
      ↓
Sigmoid
      ↓
Probability 0–1
      ↓
Threshold
      ↓
Class 0 / Class 1
```

**Logistic Regression → Classification**

**Sigmoid → Converts score to probability**

**`predict()` → Class**

**`predict_proba()` → Probability**

**Positive coefficient → Higher log-odds of class 1**

**Negative coefficient → Lower log-odds of class 1**

**Smaller `C` → Stronger regularization**

**L1 → Can create sparse coefficients**

**L2 → Shrinks coefficients**

**0.5 → Common threshold, not a universal requirement**

**Pipeline → Helps prevent preprocessing leakage during cross-validation**
