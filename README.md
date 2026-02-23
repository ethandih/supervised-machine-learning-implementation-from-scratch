# README Supervised Machine Learning Implementation from Scratch

This repository contains implementations of core supervised learning algorithms, with a focus on understanding optimization, model assumptions, and evaluation.

The main emphasis is on implementing models manually (without high-level ML libraries) to demonstrate algorithmic understanding and training dynamics.

---

## What this repository demonstrates

- Implementation of Linear Regression using Gradient Descent  
- Implementation of Logistic Regression using Gradient Descent  
- Optimization using Binary Cross-Entropy Loss  
- Feature engineering for non-linearly separable classification problems  
- Model evaluation using accuracy, confusion matrix, and ROC-AUC  
- Decision Tree and Random Forest baselines using scikit-learn  
- Handling imbalanced classification problems  
- Practical regression modeling with RMSLE evaluation  
- Basic ensembling using gradient boosting (CatBoost)  
- Data preprocessing, feature transformation, and distribution analysis  

---

## Repository Structure


```text
├── supervised_learning_main.ipynb
├── linear_regression_manual_implementation.py
├── logistic_regression_manual_implementation.py
├── data/
│   ├── mission1.csv
│   ├── mission2.csv
│   ├── mission3_train.csv
│   ├── mission3_test.csv
│   ├── final_mission_train.csv
│   └── final_mission_test.csv
└── README.md

```


## Implemented Tasks

### 1. Linear Regression (from scratch)

- Manual implementation of gradient descent  
- Derivation of fitted linear model  
- Error distribution analysis  
- Discussion of unbiased estimators  

---

### 2. Logistic Regression (from scratch + feature engineering)

- Manual implementation using gradient descent  
- Binary cross-entropy optimization  
- Feature engineering to solve non-linearly separable classification  
- Training curve visualization  
- Comparison with Decision Tree classifier  

---

### 3. Decision Trees & Imbalanced Classification

- Feature transformation to reveal hidden signal  
- Hyperparameter tuning  
- Evaluation using ROC-AUC  
- Comparison to Random Forest  

---

### 4. Regression with Real-World Data Considerations

- Exploratory data analysis  
- Train/test distribution comparison  
- Log-transform for RMSLE optimization  
- Gradient boosting using CatBoost  
- Discussion of evaluation metrics and generalization  

---

## How to Run

This project was developed as a Jupyter Notebook submission (as required by the course).

1. Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost
```

2. Open the notebook:

`supervised_learning_notebook.ipynb`


---

## Notes on Evaluation

- Training metrics are monitored during optimization.
- Test data is used only for final evaluation.
- ROC-AUC is used for imbalanced classification.
- RMSLE is used for regression tasks spanning multiple orders of magnitude.

---

## Why implement from scratch?

Although production systems rely on established ML libraries, implementing models manually:

- Strengthens understanding of optimization dynamics  
- Clarifies inductive bias and model assumptions  
- Improves debugging and interpretability  
- Provides insight into convergence behavior  

---

## Disclaimer

This project originated from coursework in an introductory machine learning course.  
The repository has been refactored for clarity.


