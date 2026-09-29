# Unsupervised Learning

## Machine Learning

---

## Overview

Unsupervised Learning is a type of Machine Learning where the model works with **data that does not have predefined labels or correct answers**.

The model tries to find patterns, groups, relationships, or unusual data points by itself.

For example, if we have customer information but no customer categories, an unsupervised learning algorithm can group customers based on their similarities.

```text
Unlabeled Data
      ↓
Unsupervised Algorithm
      ↓
Find Patterns
      ↓
Groups / Relationships / Anomalies
```

This section contains my notes and practice work on the main concepts and algorithms used in Unsupervised Learning.

---

# How Unsupervised Learning Works

In supervised learning, we have:

```text
Input → Correct Answer
```

In unsupervised learning, we only have:

```text
Input Data
    ↓
Find Hidden Patterns
```

Example:

```text
Age    Income
22     20000
24     25000
23     22000
45     90000
48     95000
46     85000
```

There are no predefined labels such as:

```text
Young Customer
High Income Customer
```

The algorithm can find groups based on similarities in the data.

---

# Main Types of Unsupervised Learning

Common types include:

1. **Clustering**
2. **Dimensionality Reduction**
3. **Association Rule Learning**
4. **Anomaly Detection**

---

# 1. Clustering

Clustering is used to divide data into groups based on similarity.

Each group is called a **cluster**.

Example:

```text
Customer Data
      ↓
   Clustering
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
Group 1  Group 2  Group 3
```

The important point is that the groups are created by the algorithm rather than being given beforehand.

---

## K-Means Clustering

K-Means is one of the most commonly used clustering algorithms.

It divides data into a specified number of clusters.

For example:

```text
K = 3
```

means we want the algorithm to create 3 clusters.

Basic idea:

```text
Data Points
    ↓
Choose K
    ↓
Create Cluster Centers
    ↓
Assign Data Points
    ↓
Update Centers
    ↓
Repeat
    ↓
Final Clusters
```

Example:

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)

labels = model.labels_
```

---

## Choosing the Value of K

Choosing the correct number of clusters is important.

One common method is the **Elbow Method**.

We calculate the clustering error for different values of K and look for an elbow point.

```text
Error
 │\
 │ \
 │  \
 │   \__
 │      \__
 └────────────
       K
```

The point where the improvement starts slowing down can help us choose K.

---

## Hierarchical Clustering

Hierarchical clustering creates a hierarchy of groups.

It can be represented using a **dendrogram**.

```text
Data
 ↓
Small Groups
 ↓
Larger Groups
 ↓
Final Hierarchy
```

It can be useful when we want to understand how data points or groups are related.

---

## DBSCAN

DBSCAN stands for:

**Density-Based Spatial Clustering of Applications with Noise**

It groups data points based on their density.

One useful feature of DBSCAN is that it can identify some points as **noise or outliers**.

Example:

```text
● ● ●
 ● ●

          ×

                 ● ●
                ● ● ●
```

Here `×` can represent a point treated as noise.

---

# 2. Dimensionality Reduction

Dimensionality reduction means reducing the number of features while trying to keep important information.

Suppose a dataset has:

```text
100 Features
```

We may reduce it to:

```text
2 or 3 Features
```

This can make data easier to visualize and can sometimes reduce computational complexity.

```text
High-Dimensional Data
          ↓
Dimensionality Reduction
          ↓
Low-Dimensional Data
```

---

## PCA

PCA stands for **Principal Component Analysis**.

It transforms the original features into a smaller number of new features called **principal components**.

The first principal component tries to capture the largest amount of variation in the data, followed by the next components.

Example:

```text
50 Features
     ↓
    PCA
     ↓
2 Components
     ↓
Visualization
```

Simple example:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)
```

---

## Why Use PCA?

PCA can be useful for:

* Reducing the number of features
* Visualizing high-dimensional data
* Removing some redundancy
* Making some models faster
* Understanding major patterns in data

PCA does not simply select existing columns. It creates new components from the original features.

---

# 3. Association Rule Learning

Association rule learning is used to find relationships between items.

It is commonly used in **market basket analysis**.

Example:

```text
Customer buys:
Bread + Butter
```

The algorithm may discover that these products frequently occur together.

Example:

```text
Bread
  +
Butter
  ↓
Frequently Purchased Together
```

---

## Apriori Algorithm

Apriori is a popular algorithm for finding frequent item combinations and association rules.

Example:

```text
Milk → Bread
```

This can represent a rule where customers who buy milk also frequently buy bread.

Important terms:

### Support

Support tells us how frequently an item or item combination appears in the dataset.

### Confidence

Confidence tells us how often the consequent occurs when the antecedent occurs.

### Lift

Lift compares the observed relationship with what would be expected if the items were independent.

A higher lift can indicate a stronger association.

---

# 4. Anomaly Detection

Anomaly detection is used to find data points that are significantly different from normal patterns.

An anomaly can also be called:

* Outlier
* Unusual observation
* Abnormal data point

Example:

```text
Normal Transactions
₹500
₹700
₹450
₹800
₹600

        ₹95,000
           ↑
        Anomaly
```

Possible applications:

* Fraud detection
* Network security
* Machine monitoring
* Sensor monitoring
* Unusual transactions
* Error detection

---

## Isolation Forest

Isolation Forest is an algorithm commonly used for anomaly detection.

It tries to isolate unusual observations from normal observations.

Example:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)

model.fit(X)

predictions = model.predict(X)
```

The result can identify observations as normal or anomalous.

---

# Unsupervised Learning vs Supervised Learning

| Supervised Learning           | Unsupervised Learning                   |
| ----------------------------- | --------------------------------------- |
| Uses labeled data             | Uses unlabeled data                     |
| Target variable is available  | No predefined target                    |
| Learns from known answers     | Finds patterns by itself                |
| Used for prediction           | Used for pattern discovery              |
| Regression and Classification | Clustering and Dimensionality Reduction |

Example:

```text
Supervised:
House Data → Known Price → Predict Price

Unsupervised:
Customer Data → No Labels → Find Customer Groups
```

---

# Data Preparation

Unsupervised algorithms can also require data preprocessing.

## Handling Missing Values

Missing values may need to be:

* Removed
* Filled with mean
* Filled with median
* Filled with another suitable value

---

## Feature Scaling

Feature scaling is especially important for distance-based algorithms such as K-Means and DBSCAN.

For example:

```text
Age       → 20 - 60
Income    → 20,000 - 200,000
```

Income has a much larger numerical range.

Without scaling, it can have a much stronger effect on distance calculations.

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# Train-Test Split

Unlike supervised learning, unsupervised learning usually does not have a target variable to split into `X` and `y`.

For many unsupervised tasks, we work directly with the available feature data.

However, separate validation or test procedures can still be used when evaluating a particular unsupervised method.

---

# Evaluating Unsupervised Learning

Evaluation is different because there may be no correct labels.

Some commonly used measures include:

### Silhouette Score

Measures how well a data point fits within its own cluster compared with other clusters.

A higher silhouette score generally indicates better-separated clusters.

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)

print(score)
```

### Inertia

In K-Means, inertia measures the sum of squared distances between data points and their assigned cluster centers.

It is commonly used with the Elbow Method.

### Visualization

For some problems, visualization can help us understand whether the discovered groups make sense.

---

# Common Algorithms

## Clustering

* K-Means
* Hierarchical Clustering
* DBSCAN

## Dimensionality Reduction

* PCA
* t-SNE
* UMAP

## Association Rules

* Apriori
* FP-Growth

## Anomaly Detection

* Isolation Forest
* One-Class SVM

---

# Applications

Unsupervised Learning is used in many real-world problems.

### Customer Segmentation

```text
Customer Data
      ↓
Clustering
      ↓
Customer Groups
```

### Market Basket Analysis

```text
Shopping Data
      ↓
Association Rules
      ↓
Related Products
```

### Fraud Detection

```text
Transaction Data
      ↓
Anomaly Detection
      ↓
Unusual Transactions
```

### Data Visualization

```text
High-Dimensional Data
      ↓
PCA
      ↓
2D / 3D Representation
```

### Image Analysis

Clustering and dimensionality reduction can also be used to discover patterns in image data.

---

# Tools Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* Jupyter Notebook
* VS Code

---

# Learning Outcomes

After studying Unsupervised Learning, I can:

* Understand what unsupervised learning means
* Understand labeled and unlabeled data
* Explain clustering
* Understand K-Means
* Understand Hierarchical Clustering
* Understand DBSCAN
* Understand dimensionality reduction
* Understand PCA
* Understand association rules
* Understand Apriori
* Understand anomaly detection
* Understand Isolation Forest
* Prepare data for unsupervised algorithms
* Apply feature scaling
* Use clustering evaluation metrics
* Visualize discovered patterns

---

# Interview Questions

### Basic Questions

**1. What is Unsupervised Learning?**

Unsupervised Learning is a type of Machine Learning where the model learns patterns from data without predefined target labels.

**2. What is the difference between supervised and unsupervised learning?**

Supervised learning uses labeled data, while unsupervised learning works mainly with unlabeled data to discover patterns.

**3. What is clustering?**

Clustering is the process of grouping similar data points together.

**4. Give some examples of clustering algorithms.**

K-Means, Hierarchical Clustering, and DBSCAN.

**5. What is an unlabeled dataset?**

A dataset where the expected output or target class is not provided.

---

### Clustering Questions

**6. What is K-Means?**

K-Means is a clustering algorithm that divides data into a specified number of clusters.

**7. What does K represent in K-Means?**

K represents the number of clusters we want to create.

**8. What is the Elbow Method?**

The Elbow Method helps choose a suitable value of K by observing how clustering error changes for different values of K.

**9. What is DBSCAN?**

DBSCAN is a density-based clustering algorithm that can identify dense groups and noise points.

**10. What is the difference between K-Means and DBSCAN?**

K-Means requires the number of clusters beforehand, while DBSCAN uses density and can identify some points as noise.

---

### Dimensionality Reduction Questions

**11. What is dimensionality reduction?**

It is the process of reducing the number of features while trying to preserve important information.

**12. What is PCA?**

PCA is a dimensionality reduction technique that transforms original features into principal components.

**13. Why is PCA used?**

PCA can reduce the number of features, help visualize data, and remove some redundancy.

---

### Other Questions

**14. What is anomaly detection?**

Anomaly detection identifies data points that are significantly different from normal observations.

**15. What is Isolation Forest?**

Isolation Forest is an algorithm used to identify unusual or anomalous observations.

**16. What is association rule learning?**

It is a technique used to find relationships between items or events that frequently occur together.

**17. What is the Apriori algorithm?**

Apriori is an algorithm used to find frequent itemsets and association rules.

**18. What is Support?**

Support measures how frequently an item or item combination occurs in the dataset.

**19. What is Confidence?**

Confidence measures how often the consequent occurs when the antecedent occurs.

**20. What is Lift?**

Lift measures how much stronger an association is compared with what would be expected if the items were independent.

---

# Practical Examples

### Customer Segmentation

```text
Customer Data
      ↓
K-Means
      ↓
Customer Groups
```

### Data Visualization

```text
Many Features
      ↓
PCA
      ↓
2 Features
      ↓
Visualization
```





**Part of AI Forge | Machine Learning**

