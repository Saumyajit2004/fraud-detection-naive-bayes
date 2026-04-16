# Credit Card Fraud Detection (Naive Bayes)

## Overview
This project focuses on detecting fraudulent credit card transactions using the Gaussian Naive Bayes model on a highly imbalanced dataset (~0.17% fraud rate).

## Note
This work was part of a team project where multiple ML models were compared.  
This repository contains **only my contribution (Naive Bayes model)**.

## My Contribution
- Implemented **Gaussian Naive Bayes** for fraud detection  
- Applied **SMOTE** for handling class imbalance  
- Used **class weighting and threshold tuning** to improve performance  
- Achieved **~87% recall**, effectively detecting most fraudulent transactions  

## Results
| Metric    | Value |
|----------|------|
| Recall   | ~87% |
| Precision| Low  |
| F1 Score | ~0.18 |

## Key Insight
Naive Bayes achieves high recall but suffers from **high false positives** due to the independence assumption, making it less suitable for real-world deployment compared to ensemble models.

## Tech Stack
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Imbalanced-learn  

## Dataset
Credit Card Fraud Detection dataset (highly imbalanced)
