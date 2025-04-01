# Breast Cancer Prediction

This Colab notebook explores various machine learning models for predicting breast cancer diagnosis based on features extracted from cell nuclei images.

## Data Description

The dataset used in this analysis contains 30 features computed for each cell nucleus, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.  Mean, standard error, and "worst" (mean of the three largest values) are computed for each feature, resulting in a total of 30 features. The target variable is the diagnosis: 'M' for malignant and 'B' for benign.


## Data Preprocessing

1. **Data Cleaning**: Removed the 'id' and 'Unnamed: 32' columns as they are not relevant for prediction.
2. **Feature Scaling**:  Standardized the features using StandardScaler to ensure that features with larger values do not dominate the models.
3. **Target Variable Encoding**: Replaced 'M' with 1 and 'B' with 0 for binary classification.

## Exploratory Data Analysis (EDA)

- **Class Distribution**: Visualized the proportion of malignant and benign cases using a pie chart.
- **Correlation Analysis**:  Examined the correlation between the diagnosis and various features using heatmaps. Separate heatmaps were generated for features based on their measurement type (mean, standard error, worst).


## Model Building

Multiple classification models were trained and evaluated on the dataset:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost


Model performance was evaluated using accuracy, F1-score, precision, recall, and ROC AUC.


## Artificial Neural Network (ANN)

An Artificial Neural Network was also built for comparison with the traditional machine learning models. The ANN architecture comprises multiple dense layers with ReLU activation functions, dropout layers to prevent overfitting, and a final sigmoid layer for binary classification. Model training utilized the Adam optimizer and binary cross-entropy loss. Early stopping was implemented to prevent overfitting. The model's learning process is visualized through a plot of training and validation loss.


## Results

The results of all models are summarized in terms of the above mentioned metrics. Confusion matrices are generated for both the Logistic Regression and ANN models to visualize the performance and the error types produced by them.

## Further Improvements

- Hyperparameter tuning for all models to optimize performance.
- Explore feature engineering to create new features that might be more informative.
- Try other machine learning models not included in this current notebook.
- Perform more detailed EDA including visualization of individual features and their relationship with the target variable.

