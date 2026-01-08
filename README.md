# Machine Learning for Beginners - Google Colab Notebooks

Welcome to the comprehensive Machine Learning learning path! This collection of Google Colab notebooks will take you from ML basics to implementing complete capstone projects.

## Learning Path Overview

```
START HERE
    |
    v
[01] ML Basics Introduction
    |
    +---> [02] Regression Algorithms
    |         |
    |         v
    |     [05] Capstone: House Price Prediction
    |
    +---> [03] Classification Algorithms
    |         |
    |         v
    |     [06] Capstone: Customer Churn Prediction
    |
    +---> [04] Clustering Algorithms
              |
              v
          [07] Capstone: Customer Segmentation
```

## Notebooks

### Foundation
| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_ML_Basics_Introduction.ipynb` | ML fundamentals, Python libraries, data loading, EDA basics |

### Algorithm Tutorials
| # | Notebook | Algorithms Covered |
|---|----------|-------------------|
| 02 | `02_Regression_Algorithms.ipynb` | Linear, Polynomial, Ridge, Lasso, Decision Tree, Random Forest |
| 03 | `03_Classification_Algorithms.ipynb` | Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes |
| 04 | `04_Clustering_Algorithms.ipynb` | K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models |

### Capstone Projects
| # | Notebook | Problem Type | Business Application |
|---|----------|--------------|---------------------|
| 05 | `05_Capstone_Regression_House_Price.ipynb` | Regression | Real Estate Pricing |
| 06 | `06_Capstone_Classification_Customer_Churn.ipynb` | Classification | Customer Retention |
| 07 | `07_Capstone_Clustering_Customer_Segmentation.ipynb` | Clustering | Marketing Strategy |

## How to Use These Notebooks

### Option 1: Google Colab (Recommended)
1. Open Google Colab: https://colab.research.google.com
2. File -> Upload notebook
3. Select the `.ipynb` file
4. Run cells sequentially (Shift + Enter)

### Option 2: Local Jupyter
```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
jupyter notebook
```

## Recommended Dataset Sources

### Kaggle
- **House Prices**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Telco Churn**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Mall Customers**: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **Iris Dataset**: https://www.kaggle.com/datasets/uciml/iris

### HuggingFace Datasets
```python
from datasets import load_dataset

# Example datasets
dataset = load_dataset("scikit-learn/iris")
dataset = load_dataset("mstz/heart_failure")
```

### Scikit-learn Built-in
```python
from sklearn.datasets import (
    load_iris,           # Classification
    load_wine,           # Classification
    load_breast_cancer,  # Classification
    fetch_california_housing,  # Regression
    load_diabetes        # Regression
)
```

## Key Concepts by Notebook

### 01 - ML Basics
- What is Machine Learning?
- Types: Supervised, Unsupervised, Reinforcement
- Essential Libraries: NumPy, Pandas, Matplotlib, Scikit-learn
- ML Workflow: Problem -> Data -> EDA -> Preprocess -> Model -> Evaluate
- Loading datasets from various sources

### 02 - Regression
- Predicting continuous values
- Metrics: MAE, MSE, RMSE, R^2
- Algorithms:
  - Linear Regression (baseline)
  - Polynomial Regression (non-linear)
  - Ridge/Lasso (regularization)
  - Decision Tree/Random Forest (tree-based)

### 03 - Classification
- Predicting categories
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion Matrix interpretation
- Handling imbalanced data
- Algorithms comparison

### 04 - Clustering
- Unsupervised learning
- Metrics: Silhouette, Inertia, Davies-Bouldin
- Finding optimal k (Elbow method)
- Algorithm strengths and weaknesses

### 05-07 - Capstone Projects
- End-to-end ML workflow
- Feature engineering
- Model comparison and selection
- Hyperparameter tuning
- Business recommendations

## Prerequisites

### Knowledge
- Basic Python programming
- Understanding of variables, loops, functions
- Basic statistics (mean, median, standard deviation)

### Software
```bash
# Core libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Additional for capstones
pip install xgboost imbalanced-learn

# For HuggingFace datasets
pip install datasets
```

## Capstone Project Guidelines

### Documentation Requirements
Each capstone project should include:

1. **Problem Statement**
   - Clear objective
   - Business context
   - Success metrics

2. **Data Understanding**
   - Data source
   - Feature descriptions
   - Initial statistics

3. **Methodology**
   - Data preprocessing steps
   - Feature engineering decisions
   - Model selection rationale

4. **Results**
   - Model performance metrics
   - Visualizations
   - Feature importance

5. **Conclusions**
   - Key findings
   - Business recommendations
   - Future improvements

### Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Code Quality | 20% |
| EDA Depth | 20% |
| Model Performance | 25% |
| Documentation | 20% |
| Business Insights | 15% |

## Quick Reference

### Common Import Block
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
```

### ML Workflow Template
```python
# 1. Load Data
df = pd.read_csv('data.csv')

# 2. EDA
df.info()
df.describe()

# 3. Preprocess
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model
model = SomeModel()
model.fit(X_train_scaled, y_train)

# 6. Evaluate
predictions = model.predict(X_test_scaled)
score = some_metric(y_test, predictions)
```

## Troubleshooting

### Common Issues

1. **Memory Error in Colab**
   - Use smaller dataset sample
   - Clear outputs: Edit -> Clear all outputs
   - Restart runtime: Runtime -> Restart runtime

2. **Package Not Found**
   ```python
   !pip install package_name
   ```

3. **Slow Training**
   - Reduce dataset size for testing
   - Use `n_jobs=-1` for parallel processing
   - Consider using GPU runtime in Colab

## Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new examples

## Resources

### Online Courses
- Coursera: Machine Learning by Andrew Ng
- Fast.ai: Practical Deep Learning

### Books
- "Hands-On Machine Learning" by Aurelien Geron
- "Python Machine Learning" by Sebastian Raschka

### Documentation
- Scikit-learn: https://scikit-learn.org/stable/
- Pandas: https://pandas.pydata.org/docs/
- Matplotlib: https://matplotlib.org/stable/

---

Happy Learning!
