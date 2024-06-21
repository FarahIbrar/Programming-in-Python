# Breast Cancer Wisconsin Classification

## Description
This project involves building and comparing classification models using two different algorithms: Decision Tree and Logistic Regression. The dataset used is the "Breast Cancer Wisconsin" dataset, obtained from the UCI Machine Learning Repository. The goal is to predict whether a breast tumor is malignant or benign based on various features.

## What is Wisconsin Classification?
- Wisconsin Classification refers to a method or system of categorizing or classifying data, typically used in the context of machine learning and statistical analysis. It involves grouping data points into distinct classes or categories based on their attributes or characteristics.
- The term "Wisconsin Classification" specifically originates from the dataset known as the "Breast Cancer Wisconsin (Diagnostic) Dataset," where machine learning algorithms are applied to classify breast cancer tumors as either malignant or benign based on various features extracted from cell images (exactly the same dataset found in this respository for scikit-learn).
- The classification process aims to assist in medical diagnostics by predicting the nature of tumors, thereby aiding in treatment decisions and patient care strategies.

## Aim
The aim of this project is to develop machine learning models that can accurately classify breast tumors as either malignant or benign using the provided dataset. By comparing the performance of Decision Tree and Logistic Regression algorithms, we aim to determine which model is more suitable for this classification task.

## Need for the Project
Breast cancer is a significant health concern globally, and early detection plays a crucial role in improving patient outcomes. Machine learning models can assist in automating and improving the accuracy of diagnosis based on medical data. This project serves to demonstrate the application of classification algorithms in medical diagnostics and to provide insights into model selection for similar tasks.

## Source of the Data
The Breast Cancer Wisconsin Classification dataset was sourced from the UCI Machine Learning Repository.

## Steps Involved and Why They Were Needed
1. **Importing the Dataset:**
   - **Purpose:** Obtain the dataset containing various features related to breast tumor characteristics.
   
2. **Building Classification Models:**
   - **Decision Tree:**
     - **Purpose:** Decision Trees are interpretable models that can reveal insights into the decision-making process of classifying tumors.
   - **Logistic Regression:**
     - **Purpose:** Logistic Regression provides probabilistic outputs and is effective for linearly separable data, helping to predict tumor malignancy with probabilities.
   
3. **Model Evaluation and Comparison:**
   - **Purpose:** Compare the performance metrics (accuracy, precision, recall, F1-score) of Decision Tree and Logistic Regression models to determine which one is more effective for this specific classification task.
   
## Results
- **Decision Tree:**
  - Precision: 0.93 (class 0), 0.96 (class 1)
  - Recall: 0.93 (class 0), 0.96 (class 1)
  - F1-score: 0.93 (class 0), 0.96 (class 1)
  - Accuracy: 0.95

- **Logistic Regression:**
  - Precision: 0.98 (class 0), 0.97 (class 1)
  - Recall: 0.95 (class 0), 0.99 (class 1)
  - F1-score: 0.96 (class 0), 0.98 (class 1)
  - Accuracy: 0.97

## Discussion
The primary goal of this project was to predict whether a breast tumor is malignant or benign based on various features. The models developed, Decision Tree and Logistic Regression, were evaluated based on their performance metrics on the test dataset. 
Not only this but the decision to recommend Logistic Regression over Decision Tree was based on its superior performance metrics on the breast cancer dataset. Logistic Regression not only provided higher accuracy but also demonstrated better precision and recall, which are critical for medical diagnostics where correctly identifying malignant cases is crucial.

## Model Performance Comparison:
- **Decision Tree:** Achieved an accuracy of 0.95, with precision and recall scores of 0.93 and 0.96 for class 0 (benign) and class 1 (malignant), respectively.
- **Logistic Regression:** Demonstrated higher performance with an accuracy of 0.97. It showed precision and recall scores of 0.98 and 0.97 for class 0, and 0.95 and 0.99 for class 1, respectively.
    - Outperforms the Decision Tree model in terms of accuracy, precision, recall, and F1-score for both classes.
    - It is recommended for this classification task due to its higher predictive accuracy and robust performance across evaluation metrics.
      
## Implications for Breast Cancer Diagnosis:
The decision to recommend Logistic Regression over Decision Tree was based on its superior performance metrics on the breast cancer dataset. Logistic Regression not only provided higher accuracy but also demonstrated better precision and recall, which are critical for medical diagnostics where correctly identifying malignant cases is crucial.

## Practical Applications:
- **Logistic Regression:** Its ability to provide probabilistic outputs and interpretability makes it suitable for clinical settings where understanding the confidence level of predictions is crucial.
- **Decision Tree:** While interpretable, it may require careful tuning to avoid overfitting and ensure reliable generalization to new data.

### Conclusion
Based on the results, **Logistic Regression** is recommended for predicting breast tumor malignancy due to its superior performance metrics and potential for practical clinical applications. However, further validation and possibly combining multiple models or ensemble methods could enhance prediction accuracy and robustness in real-world scenarios.

## What Did I Learn
Through this project:
- I learned to import and preprocess datasets using scikit-learn.
- Explored the application of Decision Tree and Logistic Regression algorithms for classification tasks.
- Gained insights into evaluating and comparing model performance using classification metrics.
- Understood the importance of selecting appropriate algorithms based on dataset characteristics and task requirements.
