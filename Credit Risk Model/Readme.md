# German Credit Risk Classification

This project aims to classify credit applications as 'Good Credit' or 'Bad Credit' using the German Credit Data dataset. Various machine learning models are explored and evaluated to determine the best approach for this imbalanced binary classification problem.

## Dataset

The dataset used in this project is the German Credit Data from the UCI Machine Learning Repository. It contains 1000 instances with 20 features and a target variable indicating the credit risk (Good or Bad).

## Project Structure

The notebook follows these steps:

1.  **Data Loading**: Load the German Credit Data into a pandas DataFrame.
2.  **Data Preprocessing**:
    *   Map the target variable to 0 (Bad Credit) and 1 (Good Credit).
    *   Separate features (X) and target (y).
    *   Handle categorical features using One-Hot Encoding.
    *   Handle numerical features using scaling (MinMaxScaler and StandardScaler were explored).
    *   Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
3.  **Model Training**: Train several classification models on the preprocessed and resampled training data, including:
    *   Logistic Regression
    *   Decision Tree
    *   Random Forest
    *   Gradient Boosting
    *   AdaBoost
    *   XGBoost
4.  **Model Evaluation**: Evaluate the trained models using various metrics on the test set:
    *   Accuracy
    *   Precision
    *   Recall (especially important for 'Bad Credit' class)
    *   F1-Score
    *   ROC-AUC Score
    *   Confusion Matrix
5.  **Feature Importance**: Visualize the most important features for the XGBoost model.

## Requirements

To run this notebook, you will need the following libraries:

*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   imblearn
*   xgboost

You can install them using pip:
