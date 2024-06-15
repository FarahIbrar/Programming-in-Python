#!/usr/bin/env python
# coding: utf-8

# # Project 1: Species Classification

# #### Step 1: Data Preprocessing

# **Detailed Steps**:
# 
# 1. **Data Preprocessing**:
#    - Clean the data, handle missing values (if any), and scale the features.
# 
# 2. **Feature Engineering**:
#    - Create new features or transformations if needed.
# 
# 3. **Model Selection**:
#    - Try out multiple algorithms (e.g., Logistic Regression, SVM, Decision Trees, Random Forests).
#    - Select the best-performing one based on evaluation metrics.
# 
# 4. **Model Evaluation**:
#    - Cross-validation, hyperparameter tuning, and evaluation using various metrics (accuracy, precision, recall, F1-score).
# 
# 5. **Interpretability**:
#    - Understand which features are most important for the classification.

# ### Step 1: Data Preprocessing
# 
# Let's start with data preprocessing. We'll clean the data, handle any missing values, and scale the features.
# 
# ```python
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_iris
# 
# # Load the Iris dataset
# iris = load_iris()
# iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_dataset['species'] = iris.target
# 
# # Display the first few rows of the dataset
# print(iris_dataset.head())
# 
# # Separate features (X) and target (y)
# X = iris_dataset.drop('species', axis=1)
# y = iris_dataset['species']
# 
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# 
# # Display the scaled features
# print("Scaled features:")
# print(pd.DataFrame(X_train_scaled, columns=X_train.columns).head())
# ```

# ### Explanation:
# 
# **Data Loading**: We load the Iris dataset using `load_iris()` function from `sklearn.datasets`.
# - **Why Needed**: Loading the data is the first step in any data analysis or machine learning task. We need the data to be in a structured format so that we can perform further operations such as preprocessing, training, and evaluation.
# 
# **Data Exploration**: We display the first few rows of the dataset to understand its structure.
# - **Why Needed**: Exploring the data helps us understand the structure, the types of features, and the distribution of values. This insight is crucial for identifying any potential issues such as missing values or outliers, and for guiding the preprocessing steps.
# 
# **Data Splitting**: We split the dataset into training and testing sets using `train_test_split()` from `sklearn.model_selection`.
# - **Why Needed**: Splitting the data ensures that we have separate datasets for training the model and for evaluating its performance. This helps in assessing how well the model generalizes to unseen data, thus preventing overfitting.
# 
# **Feature Scaling**: We scale the features using `StandardScaler()` from `sklearn.preprocessing`.
# - **Why Needed**: Scaling the features is important because many machine learning algorithms perform better when the numerical features are on the same scale. Standardization (scaling to have zero mean and unit variance) helps in improving the convergence of gradient-based algorithms and ensures that no single feature dominates due to its scale.

# 

# ### Step 2: Feature Engineering:
# 
# ```python
# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_iris
# 
# # Load the Iris dataset
# iris = load_iris()
# iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_dataset['species'] = iris.target
# 
# # Creating a new feature for petal area
# iris_dataset['petal_area'] = iris_dataset['petal length (cm)'] * iris_dataset['petal width (cm)']
# 
# # Log transformation of sepal length
# iris_dataset['sepal length (cm)_log'] = np.log(iris_dataset['sepal length (cm)'])
# 
# # Display the dataset with the new and transformed features
# print(iris_dataset.head())
# ```

# ### **Explanation**:
# - **Creating a New Feature for Petal Area**:
#   - The new feature 'petal_area' is created by multiplying the 'petal length (cm)' and 'petal width (cm)' columns. This is useful because the area of the petal can be a more informative feature than the individual length and width measurements.
#   - **Why This Step is Needed**: Adding this feature can capture more complex relationships in the data and might improve the model's performance by providing additional information that isn't immediately apparent from the original features.
# 
# - **Log Transformation of Sepal Length**:
#   - The log transformation of 'sepal length (cm)' is performed using `np.log()`. This transformation can help to normalize the distribution of the feature, reduce the effect of outliers, and stabilize the variance.
#   - **Why This Step is Needed**: Log transformation is often used to handle skewed data. It can make patterns in the data more apparent and improve the performance of many machine learning models that assume normal distribution of the input features.

# In[ ]:





# ### Step 3: Model Selection and Training
# 
# ```python
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# 
# # Splitting the dataset into features and target variable
# X = iris_dataset.drop(columns=['species'])
# y = iris_dataset['species']
# 
# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
# # Standardizing the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# 
# # Initializing the models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Support Vector Machine": SVC(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier()
# }
# 
# # Training and evaluating each model
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name} Accuracy: {accuracy:.2f}")
# ```

# ### **Explanation**:
# - **Splitting the Dataset into Features and Target Variable**:
#   - `X` contains all features except 'species', and `y` contains the target variable 'species'.
#   - **Why This Step is Needed**: Separating features and the target variable is essential for training machine learning models. The models learn from the features to predict the target variable.
# 
# - **Splitting the Data into Training and Testing Sets**:
#   - The dataset is split into 80% training and 20% testing sets using `train_test_split()`. The `stratify` parameter ensures that the class distribution in the target variable is preserved in both training and testing sets.
#   - **Why This Step is Needed**: Splitting the data allows us to train the model on one part of the data and test its performance on unseen data. This helps in evaluating the generalization ability of the model.
# 
# - **Standardizing the Features**:
#   - `StandardScaler` is used to standardize the features by removing the mean and scaling to unit variance.
#   - **Why This Step is Needed**: Standardization is important for many machine learning algorithms, especially those that rely on distance metrics (e.g., SVM, KNN). It ensures that all features contribute equally to the result and improves model performance.
# 
# - **Initializing the Models**:
#   - Four different models are initialized: Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest.
#   - **Why This Step is Needed**: Using multiple models allows us to compare their performance and select the best one for our problem.
# 
# - **Training and Evaluating Each Model**:
#   - Each model is trained on the scaled training data and evaluated on the scaled testing data. The accuracy score for each model is printed.
#   - **Why This Step is Needed**: Training the models on the training data allows them to learn the patterns in the data. Evaluating the models on the testing data helps us understand how well they generalize to new, unseen data.

# In[ ]:





# ### Step 4: Model Evaluation and Selection
# 
# ```python
# from sklearn.metrics import classification_report, confusion_matrix
# 
# # Evaluate each model with detailed metrics
# for name, model in models.items():
#     y_pred = model.predict(X_test_scaled)
#     
#     print(f"Model: {name}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#     
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print("\n")
# ```

# ### **Explanation**:
# - **Evaluating Each Model with Detailed Metrics**:
#   - For each model, we predict the target variable for the test data.
#   - **Why This Step is Needed**: Predicting the target variable for the test data allows us to evaluate the model's performance on unseen data.
# 
# - **Printing the Classification Report**:
#   - The `classification_report()` function provides detailed metrics for each class, including precision, recall, and F1-score.
#   - **Why This Step is Needed**: The classification report helps us understand the model's performance for each class. Precision, recall, and F1-score provide insights into how well the model is performing in terms of correctly identifying the classes.
# 
# - **Printing the Confusion Matrix**:
#   - The `confusion_matrix()` function provides a matrix that shows the counts of true positive, true negative, false positive, and false negative predictions.
#   - **Why This Step is Needed**: The confusion matrix gives a detailed view of how the model's predictions compare to the actual labels. It helps identify specific areas where the model may be making errors, such as confusing one class with another.

# In[ ]:





# ### Step 5: Hyperparameter Tuning
# 
# ```python
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# 
# # Define the models
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Random Forest': RandomForestClassifier(),
#     'SVM': SVC()
# }
# 
# # Define hyperparameters for tuning
# param_grid = {
#     'Logistic Regression': {
#         'C': [0.1, 1, 10],
#         'solver': ['liblinear', 'lbfgs']
#     },
#     'Random Forest': {
#         'n_estimators': [50, 100, 200],
#         'max_features': ['sqrt'],  # Changed from 'auto' to 'sqrt'
#         'max_depth': [None, 10, 20, 30]
#     },
#     'SVM': {
#         'C': [0.1, 1, 10],
#         'gamma': [1, 0.1, 0.01],
#         'kernel': ['rbf', 'linear']
#     }
# }
# 
# # Perform Grid Search CV for each model
# best_models = {}
# for name, model in models.items():
#     print(f"Tuning hyperparameters for {name}")
#     grid = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=5, scoring='accuracy')
#     grid.fit(X_train_scaled, y_train)
#     
#     best_models[name] = grid.best_estimator_
#     print(f"Best parameters for {name}: {grid.best_params_}")
#     print(f"Best cross-validated accuracy for {name}: {grid.best_score_}")
#     print("\n")
# 
# ```

# ### Explanation:
# **Tuning Hyperparameters for Each Model**:
# - For each model (Logistic Regression, Random Forest, SVM), we perform a Grid Search Cross-Validation to find the best hyperparameters.
# - **Why This Step is Needed**: Hyperparameter tuning helps in finding the optimal settings for the model that lead to the best performance. Grid Search Cross-Validation systematically works through multiple combinations of parameter values, cross-validates each combination, and determines the best parameters that improve the model's accuracy.
# 
# **Grid Search Cross-Validation**:
# - The `GridSearchCV` function is used to perform an exhaustive search over specified parameter values for each estimator.
# - **Why This Step is Needed**: Cross-validation helps in assessing the model’s performance on different subsets of the data, providing a more robust evaluation than a simple train-test split. It helps in selecting the model that generalizes well to new, unseen data.
# 
# **Evaluating the Best Model for Each Algorithm**:
# - After tuning, we identify and store the best-performing model for each algorithm.
# - **Why This Step is Needed**: Storing the best model allows us to use it later for predictions and further evaluation. It ensures that we are working with the most optimized version of each model.

# In[ ]:





# ### Step 6: Evaluate Tuned Models on the Test Set
# 
# ```python
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# 
# # Evaluate each tuned model on the test set
# for name, model in best_models.items():
#     print(f"Evaluating {name} on the test set")
#     
#     # Predict on the test set
#     y_pred = model.predict(X_test_scaled)
#     
#     # Accuracy score
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy for {name}: {accuracy:.4f}")
#     
#     # Classification report
#     print(f"Classification report for {name}:\n{classification_report(y_test, y_pred)}")
#     
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"Confusion matrix for {name}:\n{cm}\n")
# ```

# ### **Explanation**:
# - **Predict on the Test Set**:
#   - We use the tuned models to make predictions on the test set.
#   - **Why This Step is Needed**: This step helps us evaluate how well the models generalize to new, unseen data.
# 
# - **Accuracy Score**:
#   - We calculate the accuracy score for each model on the test set.
#   - **Why This Step is Needed**: Accuracy gives a simple measure of how many instances were correctly classified by the model. It's a basic metric to understand overall performance.
# 
# - **Classification Report**:
#   - We generate a classification report that includes precision, recall, and F1-score for each class.
#   - **Why This Step is Needed**: The classification report provides a detailed evaluation of the model's performance on each class, giving insights into where the model might be underperforming.
# 
# - **Confusion Matrix**:
#   - We compute the confusion matrix for each model.
#   - **Why This Step is Needed**: The confusion matrix provides a visual representation of the true positives, false positives, true negatives, and false negatives, helping us understand specific areas where the model makes errors.

# In[ ]:





# ### Step 7: Interpret Results and Draw Conclusions
# 
# ```python
# # Summarize the results of the best models
# for name, model in best_models.items():
#     print(f"Final evaluation results for {name}:")
# 
#     # Predict on the test set
#     y_pred = model.predict(X_test_scaled)
#     
#     # Accuracy score
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.4f}")
#     
#     # Classification report
#     report = classification_report(y_test, y_pred)
#     print(f"Classification Report:\n{report}")
#     
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print(f"Confusion Matrix:\n{cm}\n")
# 
# # Overall best model based on accuracy
# best_model_name = max(best_models, key=lambda name: accuracy_score(y_test, best_models[name].predict(X_test_scaled)))
# print(f"The overall best model is: {best_model_name}")
# ```

# ### **Explanation**:
# - **Final Evaluation Results**:
#   - We summarize the evaluation results for each model, including accuracy, classification report, and confusion matrix.
#   - **Why This Step is Needed**: Summarizing results helps in comparing the performance of different models in a concise manner and aids in making an informed decision about which model to use.
# 
# - **Overall Best Model**:
#   - We determine the overall best model based on the highest accuracy score on the test set.
#   - **Why This Step is Needed**: Identifying the best performing model helps in finalizing which model to deploy or use for future predictions.

# ### Interpretation of Results:
# 1. **Accuracy**:
#    - Compare the accuracy of different models to determine which one performs the best.
#    - A higher accuracy indicates that the model correctly classified a larger proportion of instances in the test set.
# 
# 2. **Classification Report**:
#    - Analyze precision, recall, and F1-score for each class to understand the performance in more detail.
#    - **Precision**: Indicates the proportion of true positives among the predicted positives.
#    - **Recall**: Indicates the proportion of true positives among the actual positives.
#    - **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.
# 
# 3. **Confusion Matrix**:
#    - Examine the confusion matrix to identify specific areas where the model makes errors (false positives and false negatives).
#    - This can provide insights into whether the model is biased towards certain classes and if any further adjustments or improvements are needed.

# In[ ]:




