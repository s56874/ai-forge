

Logistic Regression 
Type: Supervised Learning → Classification
Main Goal: Predict a class such as loan approved or not approved by
learning a relationship between input features and a binary target.

1. Algorithm Overview
Logistic Regression is a supervised Machine Learning algorithm mainly
used for classification.

Unlike Linear Regression, which predicts a continuous numerical value,
Logistic Regression predicts the probability of a class.

For our Loan Approval dataset:

Input Features Target

Age, Salary, Years Experience, Loan Approved
Credit Score, Loan Amount

Target values:

0 → Loan Not Approved
1 → Loan Approved
Main Idea
The model first calculates a linear score and then passes that score
through the Sigmoid function to get a probability between 0 and 1.

Input Features
      ↓
Linear Score
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Threshold
      ↓
Class 0 or 1
2. Core Intuition
Suppose we want to predict whether a person will get a loan.

The model considers:

Age
Salary
Years Experience
Credit Score
Loan Amount
A person with a strong credit score and suitable income may receive a
higher predicted approval probability.

The model does not directly say:

"Loan Approved"
Instead, it first calculates a probability.

Example:

Probability = 0.82
Using the common threshold of 0.5:

0.82 >= 0.5
      ↓
Class 1
      ↓
Loan Approved
Important
Logistic Regression does not directly predict a continuous value as its
final output. It predicts a probability and converts that probability
into a class.

3. How It Works
Dataset
   ↓
Data Checking
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Learn Coefficients
   ↓
Calculate Probability
   ↓
Apply Threshold
   ↓
Make Predictions
   ↓
Evaluate Model
Step-by-step
Step 1 --- Input data
We have features X and target y.

X → Features
y → Target
For our dataset:

X →
age
salary
years_experience
credit_score
loan_amount
y →
loan_approved
Step 2 --- Model calculates a linear score
The model first calculates:

z = b0 + b1x1 + b2x2 + ... + bnxn
For our dataset:

z =
b0
+ b1(age)
+ b2(salary)
+ b3(years_experience)
+ b4(credit_score)
+ b5(loan_amount)
Step 3 --- Apply Sigmoid
The linear score is passed through the Sigmoid function.

σ(z) = 1 / (1 + e^-z)
The result is between:

0 and 1
This value is interpreted as the probability of class 1.

Step 4 --- Convert probability into class
Using a threshold of 0.5:

Probability >= 0.5
        ↓
Class 1

Probability < 0.5
        ↓
Class 0
For Loan Approval:

1 → Approved
0 → Not Approved
Step 5 --- Evaluate the model
We compare:

Actual values
      ↓
Predicted values
and calculate metrics such as:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
4. Mathematical Foundation
Linear Score
The model first calculates:

z = b0 + b1x1 + b2x2 + ... + bnxn
Where:

z = linear score

b₀ = intercept

b₁ ... bₙ = learned coefficients

x₁ ... xₙ = input features

Sigmoid Function
The Sigmoid function is:

σ(z) = 1 / (1 + e^-z)
It converts any real-valued score into a value between 0 and 1.

Examples:

z = -5 → probability close to 0
z =  0 → probability = 0.5
z =  5 → probability close to 1
Why Sigmoid?
Because classification needs a probability.

Linear score
      ↓
Sigmoid
      ↓
0 to 1 probability
5. Understanding Coefficients
Suppose one feature has a positive coefficient.

A positive coefficient generally means:

Feature increases
      ↓
Linear score tends to increase
      ↓
Probability of class 1 tends to increase
A negative coefficient generally means the opposite.

You can view coefficients using:

model.coef_
Important Interview Point
A coefficient describes a model relationship. It does not automatically
prove causation.

6. Important Hyperparameters
Logistic Regression has several important parameters.

Parameter Meaning Common Use

C Controls regularization strength Tune model complexity
max_iter Maximum optimization iterations Helps convergence
solver Optimization algorithm Controls how model is trained
penalty Type of regularization L1/L2 depending on solver

Example:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,
    max_iter=1000
)
C
C controls the inverse of regularization strength.

Smaller C
   ↓
Stronger regularization

Larger C
   ↓
Weaker regularization
max_iter
Controls the maximum number of iterations used by the optimization
algorithm.

model = LogisticRegression(max_iter=1000)
A larger value can help when the model has not converged.

solver
Common solvers include:

lbfgs
liblinear
newton-cg
sag
saga
penalty
Common regularization types include:

L1
L2
7. What Happens During model.fit()?
When we execute:

model.fit(X_train, y_train)
the model:

X_train + y_train
       ↓
Find coefficients
       ↓
Calculate linear score
       ↓
Calculate probability
       ↓
Calculate loss
       ↓
Optimize coefficients
       ↓
Store learned coefficients
After training:

model.coef_
contains the learned feature coefficients.

model.intercept_
contains the intercept.

8. Python Implementation
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
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

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression(
    max_iter=1000
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability of class 1
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", auc)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
9. Evaluation Metrics
Accuracy
Accuracy measures the proportion of predictions that are correct.

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Example:

Accuracy = 0.80
means:

80% of predictions were correct.
Important
Accuracy can be misleading when classes are highly imbalanced.

Precision
Precision answers:

Out of all cases predicted as positive, how many were actually
positive?

Precision = TP / (TP + FP)
For our project:

Out of all predicted approved loans,
how many were actually approved?
Recall
Recall answers:

Out of all actual positive cases, how many did the model correctly
identify?

Recall = TP / (TP + FN)
For our project:

Out of all actually approved loans,
how many did the model correctly identify?
F1 Score
F1 Score combines Precision and Recall.

F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
It is useful when we want a balance between precision and recall.

ROC-AUC
ROC-AUC measures how well the model separates the two classes using
predicted probabilities across thresholds.

For ROC-AUC:

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
Important
For ROC-AUC, use probability values, not only the final class
predictions.

10. Confusion Matrix
A confusion matrix contains:

                 Predicted
                0       1

Actual 0       TN      FP
Actual 1       FN      TP
True Negative --- TN
Actual:

0
Predicted:

0
Loan was not approved and the model predicted not approved.

False Positive --- FP
Actual:

0
Predicted:

1
Loan was not approved, but the model predicted approved.

False Negative --- FN
Actual:

1
Predicted:

0
Loan was approved, but the model predicted not approved.

True Positive --- TP
Actual:

1
Predicted:

1
Loan was approved and the model predicted approved.

11. predict() vs predict_proba()
predict()
Returns the final class.

y_pred = model.predict(X_test)
Example:

[1, 0, 1, 1, 0]
predict_proba()
Returns probabilities for both classes.

model.predict_proba(X_test)
Example:

[[0.20, 0.80]]
Meaning:

Class 0 → 20%
Class 1 → 80%
For Loan Approval:

80% probability of approval
Remember
predict()       → Class
predict_proba() → Probability
12. Feature Scaling
Our features have different ranges.

Example:

Age              → around 20–60
Credit Score     → around 300–850
Salary           → larger values
Loan Amount      → larger values
We use:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Important
Correct:

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Do not do:

X_test = scaler.fit_transform(X_test)
because the scaler should be fitted using training data only.

13. Data Leakage
Data leakage happens when information from the test set influences
training.

Incorrect:

Complete Dataset
      ↓
Fit preprocessing
      ↓
Train-Test Split
Better:

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
For our project:

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
This prevents the test data from being used to fit the scaler.

14. Overfitting and Underfitting
Underfitting
Training performance → Low
Testing performance  → Low
Possible causes:

Model too simple

Important features missing

Strong nonlinear relationship

Overfitting
Training performance → Very high
Testing performance  → Much lower
Possible causes:

Too much model complexity

Noise

Too many features

Data leakage

Possible solutions:

Regularization

Better feature selection

Cross-validation

More representative data

15. Regularization
Regularization helps control overly large coefficients.

Common types:

L1
L2
L1 Regularization
Can make some coefficients exactly zero.

This can perform a type of feature selection.

L2 Regularization
Shrinks coefficients toward zero.

Easy memory
L1 → Can make coefficients zero
L2 → Shrinks coefficients
16. Cross-Validation
A single train-test split can give an unstable estimate, especially when
the dataset is small.

Our dataset has only:

30 rows
So cross-validation can be useful for getting a more stable estimate.

Example:

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Scores:", scores)
print("Mean Accuracy:", scores.mean())
Basic idea
Dataset
   ↓
Fold 1 → Train / Validate
Fold 2 → Train / Validate
Fold 3 → Train / Validate
Fold 4 → Train / Validate
Fold 5 → Train / Validate
   ↓
Average performance
17. Advantages
Simple to understand.

Fast to train.

Fast to predict.

Works well as a classification baseline.

Produces probabilities.

Coefficients can be interpreted.

Supports regularization.

Works well when the decision boundary is approximately linear.

18. Disadvantages
Basic Logistic Regression assumes a linear decision boundary.

It may struggle with strongly nonlinear relationships.

Feature scaling is often useful.

Outliers can affect the fitted model.

Highly correlated features can make coefficient interpretation
difficult.

It may underfit complex datasets.

The classification threshold may need adjustment depending on the
application.

19. When NOT to Use Logistic Regression
Basic Logistic Regression may not be suitable as the main model when:

The relationship is strongly nonlinear.

Complex feature interactions are important.

The dataset requires a nonlinear decision boundary.

Another model gives better validated performance.

Possible alternatives include:

Decision Tree
Random Forest
Gradient Boosting
XGBoost
SVM with nonlinear kernel
The model should be selected using the data characteristics, validation
performance, interpretability requirements, and computational
constraints.

20. Logistic Regression vs Linear Regression
Feature Linear Regression Logistic Regression

Problem Regression Classification

Target Continuous value Class

Example House Price Loan Approval

Main output Numerical prediction Probability / class

Function Linear equation Linear equation +
Sigmoid

Common metrics MAE, RMSE, R² Accuracy, Precision,
Recall, F1, ROC-AUC

Typical target Price, Salary 0 / 1
Easy memory
House Price → Linear Regression

Loan Approval → Logistic Regression
21. Real-World Use Cases
Logistic Regression can be used for:

Loan approval prediction

Spam detection

Customer churn prediction

Disease classification

Fraud classification

Customer response prediction

Binary risk classification

22. Common Mistakes
Mistake 1 --- Using the target as a feature
Wrong:

X = df
Correct:

X = df.drop("loan_approved", axis=1)
Mistake 2 --- Forgetting the target
Correct:

y = df["loan_approved"]
Mistake 3 --- Fitting scaler on test data
Wrong:

X_test = scaler.fit_transform(X_test)
Correct:

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Mistake 4 --- Using predict() for ROC-AUC
Wrong:

roc_auc_score(y_test, y_pred)
Use probabilities:

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc_score(y_test, y_prob)
Mistake 5 --- Calling probability "accuracy"
Wrong:

Probability = 0.80
→ Model accuracy = 80%
A predicted probability and model accuracy are different things.

Mistake 6 --- Assuming 0.5 is always the best threshold
0.5 is a common default threshold, but the appropriate threshold can
depend on the application and the costs of false positives and false
negatives.

23. Practical Experiment
Use the Loan Approval dataset.

Dataset
   ↓
Check data
   ↓
Separate X and y
   ↓
Train/Test Split
   ↓
StandardScaler
   ↓
Logistic Regression
   ↓
Accuracy + Precision + Recall + F1
   ↓
Confusion Matrix
   ↓
ROC-AUC
   ↓
Cross-Validation
Experiment 1 --- Change Features
Start with:

credit_score
Then:

credit_score + salary
Then:

credit_score + salary + loan_amount
Then compare all features.

Observe how the evaluation metrics change.

Experiment 2 --- Change C
Compare:

LogisticRegression(C=0.01)
LogisticRegression(C=1)
LogisticRegression(C=100)
Observe how regularization affects the model.

Experiment 3 --- Change Threshold
Instead of using only the default 0.5, calculate probabilities:

y_prob = model.predict_proba(X_test)[:, 1]
Then compare different thresholds such as:

0.3
0.5
0.7
Observe how:

Precision
Recall
change.

24. Interview Questions
Basic
1. What is Logistic Regression?
Answer: Logistic Regression is a supervised Machine Learning
algorithm mainly used for classification. It calculates a probability
using a linear score and the sigmoid function.

2. Is Logistic Regression supervised or unsupervised?
Answer: Supervised, because it learns from labeled input-output
data.

3. Is Logistic Regression classification or regression?
Answer: It is mainly used for classification.

4. What does Logistic Regression predict?
Answer: It predicts a probability for a class and can convert that
probability into a class label.

5. What is the sigmoid function?
Answer: The sigmoid function converts the linear score into a value
between 0 and 1.

Intermediate
6. What is the formula of Logistic Regression?
Answer:

P(y=1|x) = 1 / (1 + e^-z)
where:

z = b0 + b1x1 + ... + bnxn
7. What is the meaning of a coefficient?
Answer: It describes how a feature affects the model's linear score
and therefore influences the probability of class 1, with the exact
interpretation depending on feature scaling and other model features.

8. Why do we use StandardScaler?
Answer: Our features have different numerical ranges, so scaling
puts them on a comparable scale and can make optimization more stable.

9. What is the difference between predict() and predict_proba()?
Answer:

predict() → final class
predict_proba() → probability for each class
10. What is a confusion matrix?
Answer: It shows the counts of TP, TN, FP, and FN.

Technical
11. What is Precision?
Answer:

Precision = TP / (TP + FP)
It tells us how many predicted positive cases were actually positive.

12. What is Recall?
Answer:

Recall = TP / (TP + FN)
It tells us how many actual positive cases were correctly identified.

13. What is F1 Score?
Answer: F1 Score is the harmonic mean of Precision and Recall.

14. What is ROC-AUC?
Answer: ROC-AUC measures how well the model separates the two
classes across probability thresholds.

15. What is regularization?
Answer: Regularization adds a penalty to control large coefficients
and reduce overfitting.

16. What does C do?
Answer: C controls the inverse strength of regularization.

Small C → Stronger regularization
Large C → Weaker regularization
17. Why use cross-validation?
Answer: It evaluates the model across multiple train-validation
splits and gives a more stable estimate of generalization performance.

25. Common Interview Traps
Trap 1
"Logistic Regression predicts a continuous value like Linear
Regression."

Wrong.

It is mainly used for classification and produces class probabilities.

Trap 2
"Sigmoid directly gives 0 or 1."

Not exactly.

Sigmoid gives a value between 0 and 1. A threshold is then used to
convert the probability into a class.

Trap 3
"predict() gives probability."

Wrong.

predict() → Class
predict_proba() → Probability
Trap 4
"Accuracy is the same as precision."

Wrong.

Accuracy, precision, recall, and F1 measure different aspects of model
performance.

Trap 5
"ROC-AUC should use predicted classes."

Not normally.

ROC-AUC is calculated from scores or probabilities across thresholds.

Trap 6
"C is regularization strength."

More precisely:

C is the inverse of regularization strength.
So:

Smaller C → stronger regularization
Larger C → weaker regularization
26. One-Minute Interview Explanation
"Logistic Regression is a supervised learning algorithm mainly used
for classification. In my Loan Approval project, I used age, salary,
years of experience, credit score, and loan amount as input features,
and loan_approved as the target, where 0 means not approved and 1
means approved. I split the data into training and testing sets and
scaled the numerical features using StandardScaler. During training,
Logistic Regression learns coefficients and an intercept. It
calculates a linear score and passes it through the sigmoid function
to get a probability between 0 and 1. A threshold is then used to
convert the probability into a class. I evaluated the model using
accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC."

27. Quick Revision
Algorithm:
Logistic Regression

Type:
Supervised Learning

Problem:
Classification

Our Target:
loan_approved

Target Classes:
0 → Not Approved
1 → Approved

Features:
age
salary
years_experience
credit_score
loan_amount

Main Formula:
z = b0 + b1x1 + ... + bnxn

Sigmoid:
σ(z) = 1 / (1 + e^-z)

Output:
Probability between 0 and 1

Threshold:
Common default = 0.5

Important Functions:
fit()
predict()
predict_proba()

Important Metrics:
Accuracy
Precision
Recall
F1
ROC-AUC

Important Concepts:
Sigmoid
Threshold
Confusion Matrix
Regularization
Feature Scaling
Data Leakage
Cross-Validation
Overfitting
Underfitting
28. Final Revision Flow
                 Logistic Regression
                         ↓
                Supervised Learning
                         ↓
                    Classification
                         ↓
                  Loan Approval
                         ↓
             Features + Target
                         ↓
                 Train-Test Split
                         ↓
                 Feature Scaling
                         ↓
              Learn Coefficients
                         ↓
                  Linear Score
                         ↓
                     Sigmoid
                         ↓
                  Probability
                         ↓
                    Threshold
                         ↓
                    Class 0 / 1
                         ↓
              Accuracy / Precision
              Recall / F1 / ROC-AUC
                         ↓
               Confusion Matrix
                         ↓
                Cross-Validation
                         ↓
                  Interview Ready
