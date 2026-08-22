#  Statistical Data Visualization with Seaborn

A collection of my learning notes, practice exercises, and statistical data visualization examples using **Seaborn**.

This module is part of my learning journey in **Python for Data Science, Machine Learning, and Artificial Intelligence**.

---

##  Overview

**Seaborn** is a powerful Python library for creating attractive and informative statistical visualizations.

Built on top of **Matplotlib**, Seaborn makes it easier to visualize complex datasets and works especially well with **Pandas DataFrames**.

Seaborn is commonly used in:

- Exploratory Data Analysis (EDA)
- Data Analysis
- Data Science
- Machine Learning
- Artificial Intelligence
- Business Analytics
- Statistical Analysis

---

#  Learning Objectives

After completing this module, I can:

- Create statistical visualizations.
- Analyze numerical and categorical data.
- Understand data distributions.
- Compare different categories.
- Analyze relationships between variables.
- Identify potential outliers.
- Visualize correlations.
- Use color grouping with `hue`.
- Customize Seaborn visualizations.
- Apply Seaborn to real-world datasets.

---

#  Topics Covered

## 1. Introduction to Seaborn

Learned how to use Seaborn for statistical data visualization.

### Importing Seaborn

```python
import seaborn as sns
```

Seaborn works effectively with Pandas DataFrames.

```python
import pandas as pd
```

---

## 2. Count Plot

A count plot is used to display the number of observations in different categories.

### Real-World Examples

- Number of students in each department
- Number of employees in each department
- Customer categories
- Product category analysis

### Function Used

```python
sns.countplot()
```

---

## 3. Bar Plot

A bar plot is used to compare numerical values across different categories.

### Real-World Examples

- Average salary by department
- Average marks by subject
- Average sales by product
- Average customer spending

### Function Used

```python
sns.barplot()
```

---

## 4. Scatter Plot

A scatter plot helps analyze relationships between numerical variables.

### Real-World Examples

- Experience vs salary
- Study hours vs marks
- House area vs price
- Advertising cost vs sales

### Function Used

```python
sns.scatterplot()
```

---

## 5. Line Plot

A line plot is used to visualize trends and changes over time.

### Real-World Examples

- Monthly sales
- Revenue growth
- Temperature changes
- Website traffic

### Function Used

```python
sns.lineplot()
```

---

## 6. Histogram

A histogram helps understand the distribution of numerical data.

### Real-World Examples

- Employee salary distribution
- Student marks distribution
- Customer age distribution

### Function Used

```python
sns.histplot()
```

---

## 7. Box Plot

A box plot is useful for understanding data distribution and identifying potential outliers.

### Concepts Covered

- Minimum value
- Maximum value
- Median
- Quartiles
- Data spread
- Potential outliers

### Real-World Examples

- Salary analysis
- Student marks analysis
- Customer spending analysis

### Function Used

```python
sns.boxplot()
```

---

## 8. Heatmap

A heatmap is used to visualize relationships and correlations between numerical variables.

### Real-World Examples

- Feature correlation
- Student performance analysis
- Sales analysis
- Customer behavior analysis

### Function Used

```python
sns.heatmap()
```

---

## 9. Hue

The `hue` parameter is used to compare different groups within the same visualization.

### Real-World Examples

- Gender comparison
- Department comparison
- Product categories
- Customer segments

### Example

```python
sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary",
    hue="Department"
)
```

---

## 10. Seaborn Styling

Learned how to improve the appearance of visualizations.

### Functions Used

```python
sns.set_theme()
sns.set_style()
```

---

#  Real-World Applications

Seaborn can be used for:

- Exploratory Data Analysis (EDA)
- Customer Behavior Analysis
- Employee Salary Analysis
- Student Performance Analysis
- Sales Analysis
- Healthcare Data Analysis
- Financial Analysis
- Machine Learning Feature Analysis

---

# 📊 Visualizations Covered

| Visualization | Purpose |
|---|---|
| Count Plot | Count categorical observations |
| Bar Plot | Compare numerical values by category |
| Scatter Plot | Analyze relationships |
| Line Plot | Analyze trends |
| Histogram | Understand data distribution |
| Box Plot | Identify potential outliers |
| Heatmap | Analyze correlations |

---

#  Technologies Used

- Python
- Seaborn
- Matplotlib
- Pandas
- NumPy
- Google Colab
- Jupyter Notebook
- Visual Studio Code
- Git & GitHub

---

# 💡 Key Takeaways

After completing this module, I can:

- Create statistical visualizations using Seaborn.
- Analyze numerical and categorical data.
- Understand data distributions.
- Compare categories using visualizations.
- Analyze relationships between variables.
- Identify potential outliers using box plots.
- Analyze correlations using heatmaps.
- Use the `hue` parameter for group comparison.
- Apply Seaborn to Exploratory Data Analysis.
- Use Seaborn with Pandas DataFrames.

---

##  Learning Journey

Learning **Seaborn** as part of my journey:

**Python → Data Analysis → Exploratory Data Analysis → Data Visualization → Data Science → Machine Learning → Artificial Intelligence**

⭐ *Learning by exploring data, discovering patterns, and turning raw data into meaningful insights.*
