# Credit Card Fraud Detection (Naive Bayes)

## Overview

This project focuses on detecting fraudulent credit card transactions using the **Gaussian Naive Bayes** model on a highly imbalanced dataset (~0.17% fraud rate).

## Note

This work was part of a team project where multiple machine learning models were compared.
This repository contains **only my contribution (Naive Bayes model)**.

## My Contribution

* Implemented **Gaussian Naive Bayes** for fraud detection
* Applied **SMOTE** to handle class imbalance
* Used **class weighting and threshold tuning** to improve performance
* Achieved **~88% recall**, effectively detecting most fraudulent transactions

## Results

| Metric (Fraud Class) | Value |
| -------------------- | ----- |
| Precision            | 0.06  |
| Recall               | 0.88  |
| F1 Score             | 0.11  |

## Output

![Confusion Matrix](confusion_matrix.png)

## Key Insight

Naive Bayes achieves high recall but suffers from **very high false positives**, making it impractical for real-world deployment.
This is mainly due to the **independence assumption**, which does not hold for correlated transaction features.

## Why Naive Bayes Performs Poorly

* Assumes feature independence (not valid for this dataset)
* Produces poorly calibrated probabilities
* Leads to a large number of false positives
* Less effective compared to ensemble models like Random Forest and XGBoost

## Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Imbalanced-learn
* Matplotlib

## Dataset

Credit Card Fraud Detection dataset (highly imbalanced)

## How to Run

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook

```bash
naive_bayes_model.ipynb
```
