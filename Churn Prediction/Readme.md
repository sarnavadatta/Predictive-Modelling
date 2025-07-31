## Project Overview and Purpose

This project focuses on building a machine learning model to predict customer churn for a telecommunications company. Customer churn, the act of customers discontinuing their service, is a critical challenge for telecommunication companies as it directly impacts revenue and growth. By accurately predicting which customers are likely to churn, companies can proactively implement retention strategies, thereby minimizing losses and maximizing customer lifetime value.

The primary goal of this project is to develop a predictive model that can identify potential churners based on various customer attributes and service usage patterns. This project is intended for data scientists, business analysts, and telecommunications companies seeking to leverage data-driven insights for customer retention efforts.


## Dataset Description and Source

The dataset used in this project contains information about a telecommunications company's customers and their churn status. This is a publicly available dataset commonly used for churn prediction tasks.

The dataset includes various attributes for each customer, such as:
-   **Demographic Information:** Gender, Senior Citizen status, Partner status, Dependents status.
-   **Service Information:** Phone Service, Multiple Lines, Internet Service (DSL, Fiber optic, No), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
-   **Account Information:** Contract type (Month-to-month, One year, Two year), Paperless Billing, Payment Method, Monthly Charges, Total Charges, and Tenure (the number of months the customer has stayed with the company).
-   **Churn Status:** Whether the customer churned or not.

Initial data processing steps included:
-   Converting the `TotalCharges` column from object type to numeric, handling any conversion errors by coercing invalid parsing as NaN.
-   Removing rows with missing values (NaNs) that resulted from the `TotalCharges` conversion.
-   Creating a new categorical feature called `tenure_group` by binning the `tenure` column into groups representing different periods of customer loyalty (e.g., 1-12 months, 13-24 months, etc.).

## Methodology and Approach

The following steps outline the methodology used in this notebook to build and evaluate the churn prediction models:

1.  **Data Loading and Preprocessing:**
    *   The Telco Customer Churn dataset was loaded from a CSV file into a pandas DataFrame.
    *   The `TotalCharges` column was converted to a numeric type, and any resulting missing values were dropped.
    *   A new feature, `tenure_group`, was engineered by binning the `tenure` column to categorize customers based on their length of service.

2.  **Data Transformation:**
    *   The target variable, `Churn`, was transformed into a binary numerical format, where 'Yes' was mapped to 1 and 'No' to 0.
    *   The dataset was split into features (X) and the target variable (y). The `customerID` and original `tenure` columns were excluded from the features.
    *   The data was then split into training and testing sets using a 75/25 ratio with `train_test_split`.

3.  **Feature Scaling and Encoding:**
    *   A `ColumnTransformer` was used to apply different preprocessing steps to different types of features.
    *   Categorical features were one-hot encoded using `OneHotEncoder`, which converts categorical variables into a numerical format.
    *   Numerical features were scaled using `MinMaxScaler` to bring all features into a similar range, which helps improve the performance of certain algorithms.
    *   This preprocessor was fitted on the training data and then used to transform both the training and testing datasets to prevent data leakage.

4.  **Handling Class Imbalance:**
    *   The training data exhibited a class imbalance, with fewer instances of churners. To address this, Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data.
    *   SMOTE works by creating synthetic samples from the minority class, resulting in a balanced training set (`X_train_resample`, `y_train_resample`).

5.  **Model Training and Evaluation:**
    *   Several classification models were trained on the balanced training data:
        *   Logistic Regression
        *   Decision Tree
        *   Random Forest
        *   Gradient Boosting
        *   AdaBoost
        *   XGBoost
    *   Each model's performance was evaluated on both the training and testing sets using the following metrics:
        *   **Accuracy:** The proportion of correctly classified instances.
        *   **F1-score:** The harmonic mean of precision and recall.
        *   **Precision:** The proportion of true positive predictions among all positive predictions.
        *   **Recall:** The proportion of true positive predictions among all actual positive instances.
        *   **ROC AUC Score:** The area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between classes.
## Code Structure and Usage

The notebook is structured to follow a clear and logical workflow for building and evaluating churn prediction models. The code is organized into the following sections:

1.  **Importing Libraries:** All necessary Python libraries, such as pandas, numpy, matplotlib, seaborn, and scikit-learn, are imported at the beginning of the notebook.
2.  **Data Loading and Preprocessing:** The Telco Customer Churn dataset is loaded from a CSV file. Initial data cleaning and feature engineering, such as handling missing values and creating the `tenure_group` feature, are performed in this section.
3.  **Data Splitting:** The dataset is split into training and testing sets to ensure that the models are evaluated on unseen data.
4.  **Data Transformation:** A `ColumnTransformer` is used to apply one-hot encoding to categorical features and min-max scaling to numerical features. This ensures that all data is in a suitable format for machine learning models.
5.  **Handling Class Imbalance:** The SMOTE technique is applied to the training data to address the class imbalance between churned and non-churned customers.
6.  **Model Training and Evaluation:** Several classification models are defined, trained on the preprocessed and resampled data, and then evaluated using various performance metrics.
7.  **Results Visualization:** The performance of the different models is visualized using bar charts to compare their accuracy and F1-scores.

## Results and Visualizations

The notebook evaluates several classification models for predicting customer churn. After addressing the class imbalance in the training data using SMOTE, the models were re-evaluated. Here's a summary of the key results on the test set after handling imbalance:

*   **Logistic Regression:** Achieved an accuracy of 0.7418, an F1-score of 0.7555, a precision of 0.5029, a recall of 0.7686, and a ROC AUC Score of 0.7504. This model shows a good balance between precision and recall, with a relatively high recall indicating its ability to identify a large proportion of actual churners.
*   **Decision Tree:** Showed high performance on the training set but significantly lower performance on the test set (Accuracy: 0.7355, F1-score: 0.7377, Precision: 0.4928, Recall: 0.5197, ROC AUC Score: 0.6656), suggesting overfitting.
*   **Random Forest:** Achieved an accuracy of 0.7651, an F1-score of 0.7650, a precision of 0.5492, a recall of 0.5480, and a ROC AUC Score of 0.6948.
*   **Gradient Boosting:** Performed well with an accuracy of 0.7730, an F1-score of 0.7804, a precision of 0.5515, a recall of 0.6900, and a ROC AUC Score of 0.7461. This model shows a good recall and F1-score, making it a strong candidate for identifying churners.
*   **AdaBoost:** Achieved an accuracy of 0.7463, an F1-score of 0.7594, a precision of 0.5087, a recall of 0.7620, and a ROC AUC Score of 0.7514. Similar to Logistic Regression, AdaBoost demonstrates a good recall.
*   **XGBoost:** Showed an accuracy of 0.7730, an F1-score of 0.7731, a precision of 0.5643, a recall of 0.5655, and a ROC AUC Score of 0.7058.

Considering the importance of identifying potential churners (high recall) while maintaining a reasonable balance with precision, **Logistic Regression, Gradient Boosting, and AdaBoost** appear to be among the better-performing models after handling class imbalance, based on their F1-scores and recall values on the test set.

The notebook includes **bar charts** to visually compare the Accuracy and F1-scores of the different models, both before and after applying SMOTE for handling class imbalance. These visualizations help in quickly assessing the impact of the imbalance handling technique on model performance.

## Future Work and Improvements

This project provides a solid foundation for predicting customer churn. However, there are several avenues for future work and improvements to further enhance the model's performance and practical applicability:

*   **Hyperparameter Tuning:** Systematically tune the hyperparameters of the promising models (e.g., Logistic Regression, Gradient Boosting, AdaBoost) using techniques like GridSearchCV or RandomizedSearchCV to optimize their performance on the test set.
*   **Explore Other Resampling Techniques:** Investigate other techniques for handling class imbalance, such as undersampling the majority class (e.g., RandomUnderSampler) or using hybrid approaches (e.g., SMOTEENN), and compare their impact on model performance.
*   **Advanced Feature Engineering:** Explore creating more sophisticated features from the existing data or incorporating external data sources. This could include analyzing customer interaction logs, website activity, or demographic data from external providers.
*   **Investigate Different Models:** Experiment with other machine learning algorithms suitable for classification tasks, such as Support Vector Machines (SVMs), Neural Networks, or ensemble methods tailored for imbalanced datasets.
*   **Model Interpretability:** Explore techniques for interpreting the best-performing model to understand which features are the most important drivers of churn. This can provide valuable business insights for targeted retention strategies.
*   **Model Deployment:** Develop a pipeline to deploy the trained model as a web service or integrate it into a customer relationship management (CRM) system. This would allow for real-time churn predictions and automated interventions.
*   **Monitoring and Retraining:** Implement a system to monitor the model's performance over time and set up a process for periodic retraining with new data to ensure its continued accuracy.

