# Predictive Credit Card Fraud Detection Model

## Overview

This project implements a machine learning model for detecting fraudulent credit card transactions using Python. The dataset used is obtained from Kaggle's "Credit Card Fraud Detection" dataset. The model employs `XGBoost` and `LightGBM` classifiers, with oversampling via `SMOTE` to handle class imbalance.

## Dataset

The dataset contains transactions made by credit cards in September 2013. The dataset is highly imbalanced, with fraudulent transactions constituting only a small fraction of total transactions.

## Libraries Used

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib.pyplot`: Data visualization
- `seaborn`: Statistical data visualization
- `sklearn`: Machine learning utilities
- `imblearn`: Handling imbalanced datasets
- `lightgbm`: LightGBM classifier for training
- `xgboost`: XGBoost classifier for training

## Code Breakdown

### 1. Downloading the Dataset

The dataset is downloaded from Kaggle using `kagglehub` and saved to the local directory.

### 2. Data Preprocessing

- The CSV file is read into a Pandas DataFrame.
- Missing values are checked.
- The dataset contains a highly imbalanced "Class" feature (0 for normal transactions, 1 for fraud).
- The `Amount` and `Time` columns are standardized using `StandardScaler`.

### 3. Splitting Data for Training and Testing

- The dataset is split into training (80%) and testing (20%) sets using `train_test_split()`.
- The target variable (`Class`) is separated from the features (`X`).

### 4. Handling Imbalance using SMOTE

- The training data is oversampled using `SMOTE` to balance the number of fraud and non-fraud samples.

### 5. Training an XGBoost Model

- The `XGBClassifier` is initialized with parameters optimized for handling class imbalance.
- The model is trained on the oversampled data.
- Predictions are made on the test set.
- Performance is evaluated using `classification_report` and `ROC-AUC Score`.
- A confusion matrix is plotted to visualize prediction accuracy.

### 6. Training a LightGBM Model

- A `LGBMClassifier` is initialized.
- `GridSearchCV` is used to tune hyperparameters.
- The best estimator is selected and used for predictions.
- A confusion matrix is plotted to evaluate performance.

## Model Evaluation Metrics

- **Classification Report**: Provides Precision, Recall, F1-score, and Support.
- **ROC-AUC Score**: Measures model performance in distinguishing between fraudulent and non-fraudulent transactions.
- **Confusion Matrix**: Shows actual vs predicted fraud and non-fraud counts.


This project demonstrates an end-to-end machine-learning pipeline for credit card fraud detection. By addressing data imbalance and leveraging powerful classifiers, the model effectively accurately identifies fraudulent transactions.


