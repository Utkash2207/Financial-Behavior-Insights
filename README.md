# Credit Card Fraud Detection

## Overview

This project builds a binary classification model to detect fraudulent credit card transactions. Given a dataset of labeled transactions, the model learns to distinguish between legitimate and fraudulent activity using Logistic Regression.

## Problem Statement

Credit card fraud is a significant concern for financial institutions and customers alike. The challenge is that fraudulent transactions are extremely rare compared to legitimate ones, creating a heavily imbalanced dataset. This project addresses that imbalance through under-sampling and trains a model capable of identifying fraud with high accuracy.

## Dataset

The dataset used is `creditcard.csv`, which contains anonymized credit card transaction records. Each row represents a single transaction with the following structure:

- Features V1 through V28: PCA-transformed numerical features (anonymized for confidentiality)
- Amount: the transaction amount
- Class: the target label, where 0 indicates a legitimate transaction and 1 indicates a fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions accounting for a small fraction of total records.

## Approach

**Exploratory Data Analysis**

The notebook begins by loading the dataset and inspecting its structure, checking for missing values, and examining the distribution of the target variable. Statistical summaries are computed separately for legitimate and fraudulent transactions to understand behavioral differences.

**Handling Class Imbalance**

To address the imbalance, an under-sampling strategy is applied. A random sample of 492 legitimate transactions is drawn to match the number of fraudulent transactions. The two subsets are then combined into a balanced dataset of 984 records.

**Model Training**

The balanced dataset is split into features (X) and labels (Y). A stratified train-test split is applied with 80% of the data used for training and 20% for testing. A Logistic Regression model is trained on the training set.

**Evaluation**

The model is evaluated using accuracy score on both the training and test sets.

- Training accuracy: approximately 94.5%
- Test accuracy: approximately 94.4%

The close match between training and test accuracy indicates the model generalizes well without significant overfitting.

## Project Structure

```
.
├── Credit_Card.ipynb    # Main Jupyter notebook with full analysis and model training
├── creditcard.csv       # Dataset (not included, download separately)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for all dependencies

## Installation

1. Clone or download this repository.

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle (search for "Credit Card Fraud Detection") and place `creditcard.csv` in the project directory.

4. Open and run the notebook:

```bash
jupyter notebook Credit_Card.ipynb
```

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- joblib
- threadpoolctl

## Results

The Logistic Regression model achieves approximately 94.4% accuracy on unseen test data, demonstrating that even a simple linear model can effectively identify fraudulent transactions when class imbalance is properly handled.

## Limitations

- Under-sampling discards a large portion of legitimate transaction data, which may cause the model to miss subtle patterns present in the full dataset.
- The model has not been evaluated on precision, recall, or F1-score, which are often more informative than accuracy for imbalanced classification problems.
- The features are anonymized, limiting interpretability of the model's decisions.

## License

This project is intended for educational purposes.
