# Supervised Learning

## Machine Learning

---

## Overview

Supervised Learning is one of the main types of Machine Learning.

In supervised learning, we give the model data where the **correct answer is already known**. The model studies this data, finds patterns, and then uses those patterns to make predictions for new data.

For example, if we have information about houses and their prices, we can train a model to predict the price of a new house.

```text
Input Data + Correct Answer
          ↓
       Train Model
          ↓
     Learn Patterns
          ↓
      New Data
          ↓
      Prediction
```

This section contains my notes and practice work on the main concepts and algorithms used in Supervised Learning.

---

# How Supervised Learning Works

A supervised learning dataset usually has two parts:

* **Features (`X`)** – information given to the model
* **Target (`y`)** – the value that we want the model to predict

For example:

```text
Area    Bedrooms    Bathroom    Price
1000       2           2       50 Lakh
1500       3           2       70 Lakh
2000       4           3       95 Lakh
```

Here:

```text
X = Area, Bedrooms, Bathroom
y = Price
```

The model learns the relationship between `X` and `y`.

---

# Types of Supervised Learning

Supervised Learning mainly has two types:

1. **Regression**
2. **Classification**

---

# 1. Regression

Regression is used when we want to predict a **number**.

### Examples

* House price prediction
* Salary prediction
* Sales prediction
* Temperature prediction
* Electricity consumption prediction

Example:

```text
House Details
      ↓
Regression Model
      ↓
₹67 Lakh
```

---

## Regression Algorithms

### Linear Regression

Linear Regression tries to find a relationship between input features and a numerical target.

Example:

```text
Area → House Price
```

---

### Multiple Linear Regression

Uses multiple features to make a prediction.

```text
Area + Bedrooms + Bathroom
             ↓
       House Price
```

---

### Polynomial Regression

Used when the relationship between input and target is not a simple straight line.

---

### Ridge Regression

Ridge Regression adds regularization to Linear Regression and can help reduce overfitting.

---

### Lasso Regression

Lasso Regression also uses regularization and can reduce some feature coefficients to zero.

---

### Elastic Net

Elastic Net combines Ridge and Lasso regularization.

---

### Decision Tree Regression

Uses a tree-like structure with conditions to make numerical predictions.

---

### Random Forest Regression

Uses multiple Decision Trees and combines their predictions.

---

### Support Vector Regression (SVR)

SVR is used to predict continuous numerical values using the Support Vector Machine approach.

---

### XGBoost Regression

XGBoost is a boosting algorithm that builds trees step by step to improve predictions.

---

# 2. Classification

Classification is used when the output belongs to a **category or class**.

Examples:

* Spam / Not Spam
* Fire / No Fire
* Pass / Fail
* Cat / Dog
* Positive / Negative

Example:

```text
Image
  ↓
Classification Model
  ↓
Fire
```

---

## Binary Classification

There are two classes.

```text
Fire     → 1
No Fire  → 0
```

---

## Multi-Class Classification

There are more than two classes.

```text
Cat
Dog
Horse
Bird
```

The model chooses one class as the prediction.

---

## Classification Algorithms

### Logistic Regression

Used mainly for classification problems.

It calculates probabilities and uses them to decide the class.

---

### K-Nearest Neighbors (KNN)

Looks at nearby data points and uses their classes to predict the class of a new point.

---

### Decision Tree

Uses a series of conditions to reach a final prediction.

---

### Random Forest

Combines multiple Decision Trees to make a final prediction.

---

### Naive Bayes

A probability-based algorithm commonly used for classification and text-related tasks.

---

### Support Vector Machine (SVM)

Tries to find a boundary that separates different classes.

---

### XGBoost Classification

Uses boosting with decision trees to improve classification predictions.

---

# Data Preparation

Before training a model, the dataset usually needs some preparation.

## Features and Target

```text
Features (X) → Input
Target (y)   → Answer
```

Example:

```python
X = data.drop("price", axis=1)
y = data["price"]
```

---

## Handling Missing Values

Missing values can be handled by:

* Removing rows
* Mean
* Median
* Mode

The method depends on the dataset.

---

## Categorical Data

Some datasets contain text values such as:

```text
City
Pune
Mumbai
Nagpur
```

These values may need to be converted into numerical form.

Common methods:

* Label Encoding
* One-Hot Encoding

---

## Feature Scaling

Feature scaling puts numerical features into a suitable range.

Common methods:

* Standardization
* Normalization

Scaling is especially useful for algorithms such as KNN and SVM.

---

# Train-Test Split

The dataset is usually divided into training and testing data.

```text
Complete Dataset
       ↓
 ┌─────┴─────┐
 ↓           ↓
Training    Testing
```

Training data is used to learn the patterns.

Testing data is used to check performance on unseen data.

A common split is:

```text
80% → Training
20% → Testing
```

---

# Model Training

After preparing the data, we train the model.

```python
model.fit(X_train, y_train)
```

The model learns patterns from the training data.

---

# Prediction

After training:

```python
prediction = model.predict(X_test)
```

The model uses what it learned to make predictions.

---

# Model Evaluation

Different metrics are used for different problems.

## Regression

* MAE
* MSE
* RMSE
* R² Score

## Classification

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

---

# Overfitting and Underfitting

## Overfitting

The model performs very well on training data but poorly on new data.

```text
Training → Very Good
Testing  → Poor
```

## Underfitting

The model is too simple and cannot learn the important patterns.

```text
Training → Poor
Testing  → Poor
```

---

# Model Improvement

Some common ways to improve a model:

* Better quality data
* Feature engineering
* Feature selection
* Hyperparameter tuning
* Cross-validation
* Regularization
* Trying different algorithms
* More training data

---

# Tools Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Google Colab
* Jupyter Notebook
* VS Code

---

# Learning Outcomes

After studying Supervised Learning, I can:

* Understand how supervised learning works
* Identify regression and classification problems
* Separate features and targets
* Prepare datasets
* Handle missing and categorical data
* Apply feature scaling
* Split data into training and testing sets
* Train Machine Learning models
* Make predictions
* Evaluate model performance
* Understand overfitting and underfitting
* Improve model performance

---

# Interview Questions

### Basic Questions

**1. What is Supervised Learning?**

Supervised Learning is a type of Machine Learning where the model learns from labeled data.

**2. What are features and target?**

Features are the input variables used by the model. The target is the value that the model tries to predict.

**3. What are the main types of Supervised Learning?**

Regression and Classification.

**4. What is Regression?**

Regression is used to predict continuous numerical values.

**5. What is Classification?**

Classification is used to predict categories or classes.

**6. What is the difference between training and testing data?**

Training data is used to train the model, while testing data is used to evaluate the model on unseen data.

**7. What is overfitting?**

Overfitting happens when a model learns the training data too closely and performs poorly on new data.

**8. What is underfitting?**

Underfitting happens when a model is too simple to learn the important patterns in the data.

**9. Why do we split data into training and testing sets?**

To check whether the model can perform well on data it has not seen during training.

**10. What is feature scaling?**

Feature scaling changes the range of numerical features so that they are on a comparable scale.

---

### Regression Questions

**11. What is Linear Regression?**

Linear Regression finds a linear relationship between input features and a continuous target.

**12. What is the difference between Linear and Multiple Linear Regression?**

Linear Regression can use one feature, while Multiple Linear Regression uses multiple features.

**13. What is MAE?**

MAE is the average absolute difference between actual and predicted values.

**14. What is RMSE?**

RMSE is the square root of Mean Squared Error and represents prediction error in the same unit as the target.

**15. What is R² Score?**

R² Score measures how well the model explains the variation in the target.

---

### Classification Questions

**16. What is Binary Classification?**

Classification with two possible classes, such as Fire and No Fire.

**17. What is Multi-Class Classification?**

Classification where there are more than two possible classes.

**18. What is a Confusion Matrix?**

It is a table used to understand the correct and incorrect predictions made by a classification model.

**19. What is Precision?**

Precision tells us how many predicted positive cases were actually positive.

**20. What is Recall?**

Recall tells us how many actual positive cases were correctly identified.

**21. What is F1-Score?**

F1-Score combines Precision and Recall into one metric.

---

# Practical Examples

### Regression

```text
House Details
      ↓
Regression Model
      ↓
House Price
```

### Classification

```text
Image
  ↓
Classification Model
  ↓
Fire / No Fire
```

---

# Learning Order

```text
Supervised Learning
        ↓
Data Preparation
        ↓
Regression
        ↓
Classification
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Overfitting / Underfitting
        ↓
Model Improvement
        ↓
Projects
        ↓
Interview Preparation
```

---

**Part of AI Forge | Machine Learning**

