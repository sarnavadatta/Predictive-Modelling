# Loan Default Prediction

## Project Overview and Purpose

This project focuses on building a machine learning model to predict loan defaults. The primary goal is to identify factors that contribute to loan defaults and develop a predictive model that can assist financial institutions in assessing the risk of loan applications. This can help in making informed decisions, minimizing financial losses, and optimizing lending strategies.

## Dataset Description and Source

The project uses a dataset containing information about loan applications and their default status. The dataset is sourced from an Excel file named `data_loan_default.xlsx`. It includes various features such as loan amount, interest rate, borrower information, and credit history. The target variable is 'charged_off', indicating whether the loan defaulted or not.

## Methodology and Approach

The project follows a standard data science workflow:

1.  **Data Loading and Initial Exploration:** The dataset is loaded and initial checks are performed to understand its structure, identify missing values, and examine data types.
2.  **Data Preprocessing:**
    *   Handling missing values: Missing values in numerical features (`emp_length`, `revol_util`, `pub_rec_bankruptcies`, `mort_acc`) are imputed using appropriate strategies (median, mode, KNN imputer).
    *   Feature Engineering: A new feature `yr_since_earliest_cr_line` is created from the `earliest_cr_line` and `issue_d` features to represent the number of years since the earliest credit line.
    *   Feature Selection/Dropping: Irrelevant columns like `purpose` and `application_type` are dropped based on their low variance or irrelevance to the prediction task.
3.  **Train-Test Split:** The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
4.  **Data Transformation:** Numerical features are scaled using `StandardScaler`, and categorical features are encoded using `OneHotEncoder`.
5.  **Handling Class Imbalance:** The SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to address the class imbalance in the target variable.
6.  **Model Training and Evaluation:** Several classification models, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost, are trained and evaluated using metrics like accuracy, F1-score, precision, recall, and ROC AUC score.
7.  **Model Selection and Tuning:** Based on the evaluation results, the best-performing model (initially identified as XGBoost and then further explored with Gradient Boosting) is selected. Hyperparameter tuning is performed on the chosen model using `RandomizedSearchCV` to optimize its performance.
8.  **Final Model Evaluation:** The final best-fit model is evaluated on the test set, and its performance is visualized using a confusion matrix.

## Code Structure and Usage

The code is structured in a Google Colab notebook, with cells organized logically for each step of the data science workflow.

-   **Setup and Imports:** Necessary libraries are imported.
-   **Data Loading and Initial Inspection:** The dataset is loaded from Google Drive, and initial data exploration is performed.
-   **Data Cleaning and Preprocessing:** Steps for handling missing values, feature engineering, and dropping columns are implemented.
-   **Train-Test Split and Data Transformation:** The data is split, scaled, and encoded.
-   **Handling Class Imbalance:** SMOTE is applied to the training data.
-   **Model Training and Evaluation:** Different models are trained and their performance metrics are calculated and printed.
-   **Model Visualization:** A bar plot is generated to compare the accuracy and F1 scores of the models.
-   **Fitting XGBoost and Gradient Boosting:** The code focuses on training and evaluating the XGBoost and Gradient Boosting models, including hyperparameter tuning for Gradient Boosting.
-   **Confusion Matrix Visualization:** Confusion matrices are plotted to visualize the performance of the selected models.

To use the code:

1.  Upload the `data_loan_default.xlsx` file to your Google Drive.
2.  Open the Google Colab notebook.
3.  Mount your Google Drive to access the dataset.
4.  Run the cells sequentially to execute the data science pipeline.

## Results and Visualizations

The project evaluates the performance of several classification models in predicting loan defaults. The results are presented in terms of various evaluation metrics. A bar plot visually compares the accuracy and F1 scores of the tested models. Confusion matrices are provided for the selected best-fit models (XGBoost and Gradient Boosting) to show the number of true positives, true negatives, false positives, and false negatives.

Based on the evaluation, the XGBoost and the hyperparameter-tuned Gradient Boosting models demonstrate promising performance in predicting loan defaults.

## Future Work and Improvements

Potential future work and improvements include:

-   **More Advanced Feature Engineering:** Explore creating more sophisticated features from the existing data, potentially incorporating external data sources.
-   **Trying Other Imputation Methods:** Investigate different methods for handling missing values and compare their impact on model performance.
-   **Exploring Other Models:** Experiment with other machine learning algorithms suitable for imbalanced classification problems, such as LightGBM or CatBoost.
-   **Further Hyperparameter Tuning:** Conduct more extensive hyperparameter tuning using techniques like GridSearchCV or Bayesian optimization.
-   **Explainable AI (XAI):** Apply XAI techniques to understand the model's predictions and identify the most influential features.
-   **Deployment:** Develop a plan for deploying the trained model to a production environment for real-time loan risk assessment.