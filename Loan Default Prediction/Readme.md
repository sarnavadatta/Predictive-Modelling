# Loan Default Prediction

## Project Overview and Purpose

This project aims to predict loan defaults using machine learning techniques. By analyzing historical loan data, the model identifies patterns and factors that contribute to loan defaults, enabling lenders to make informed decisions and mitigate risks.

## Dataset Description and Source

The dataset used in this project contains information about loan applicants, including their financial history, loan details, and repayment behavior. 

**Source:** The dataset is named `data_loan_default.xlsx` and is located in the Google Drive folder `/content/drive/My Drive/DS projects/Loan Default Prediction/`.

**Data fields:**
- `loan_amnt`: The listed amount of the loan applied for by the borrower.
- `term`: The number of payments on the loan. Values are in months and represented by integers.
- `int_rate`: Interest Rate on the loan.
- `installment`: The monthly payment owed by the borrower if the loan originates.
- `grade`: LC assigned loan grade.
- `sub_grade`: LC assigned loan subgrade.
- `emp_title`: The job title supplied by the borrower when applying for the loan.
- `emp_length`: Employment length in years.
- `home_ownership`: The home ownership status provided by the borrower during registration or obtained from the credit report.
- `annual_income`: The self-reported annual income provided by the borrower during registration.
- `verification_status`: Indicates if income was verified by LC, not verified, or source verified.
- `purpose`: A category provided by the borrower for the loan request. 
- `...`: Additional features related to the loan and the borrower.
- `charged_off`: **Target variable**. Indicates whether the loan was charged off (1) or not (0).

## Methodology and Approach

1. **Data Loading and Preprocessing:**
   - Load the dataset from Google Drive using `pandas`.
   - Handle missing values using median imputation (for continuous variables) and mode imputation (for categorical variables).
   - Transform the `earliest_cr_line` variable to calculate the years since the earliest credit line.

2. **Exploratory Data Analysis (EDA):**
   - Analyze data distributions, identify patterns, and gain insights into the dataset using descriptive statistics and visualizations.

3. **Feature Engineering:**
   - Create new features that may improve model performance (e.g., deriving new features from existing ones).
   - Select relevant features for modeling.

4. **Model Selection and Training:**
   - Choose an appropriate machine learning model (e.g., logistic regression, decision tree, random forest).
   - Split the data into training and testing sets.
   - Train the chosen model on the training data.

5. **Model Evaluation and Performance:**
   - Evaluate the trained model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
   - Analyze model performance and identify areas for improvement.

## Code Structure and Usage

The notebook is organized into sections with clear headings and comments to guide you through the process.

**To use this notebook:**
1. Upload the notebook to Google Colab.
2. Mount your Google Drive to access the dataset.
3. Ensure the dataset is located in the specified path within your Google Drive.
4. Execute each code cell sequentially.

## Results and Visualizations

The results and visualizations generated during the analysis are displayed within the notebook. This includes descriptive statistics, model performance metrics, and visual representations of data patterns.


## Future Work and Improvements

- Explore and incorporate more advanced feature engineering techniques.
- Experiment with different machine learning models and hyperparameter tuning to optimize performance.
- Deploy the model as a web application for real-time predictions.
