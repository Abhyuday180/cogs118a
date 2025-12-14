# Empirical Comparison of Supervised Learning Algorithms

This project conducts an empirical comparison of supervised classification algorithms
following the methodology of Caruana and Niculescu-Mizil (2006).

## Classifiers
- Logistic Regression
- Random Forest
- AdaBoost

## Datasets (UCI)
- Adult Income
- Bank Marketing
- Breast Cancer Wisconsin (Diagnostic)
- Heart Disease (Cleveland)

## Experimental Design
- Binary classification tasks
- Train/Test splits: 20/80, 50/50, 80/20
- 3 random trials per split
- Hyperparameters selected via cross-validation
- Evaluation metric: Classification Accuracy

## Goals
- Compare classifiers per dataset
- Analyze the effect of training data size
- Reproduce empirical trends reported in prior studies

This repository contains code, results, and the final report.
