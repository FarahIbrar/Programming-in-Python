#!/usr/bin/env python
# coding: utf-8

# # Scikit-Learn, EXERCISE - Notes

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Scikit-Learn, EXERCISE - Notes</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# <span style="color:#00008B;font-size:16px">
# 1. Import the “Breast Cancer Wisconsin” dataset(Ref: <a href ="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/"> UCI Machine learning repository </a>).<br><br>
# 2. Build classification model(s) using the following algorithms: a) Decision Tree, b) Logistic Regression. <br><br>
# 3. Which model would you recommend for use? Why so? <br><br>
# 
# </span>
# 
# </span></li></ul>
#     
# <br><br>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">ScikitLearn</td></tr></table>
# </footer>
# 
# </div>

# In[ ]:





# ### Step 1: Installing scikit-learn
# 
# **Generalized Explanation:**
# ```python
# !pip install scikit-learn
# ```
# This command installs the scikit-learn library, a popular machine learning library in Python.
# 
# **Why It Was Needed:**
# Scikit-learn provides simple and efficient tools for data mining and data analysis, making it essential for various machine learning tasks such as classification, regression, clustering, and more.
# 
# **Why Did It Show/Discussion of Results:**
# After running this command, you should see output indicating the installation process. If the installation is successful, you'll see messages confirming that scikit-learn and its dependencies have been installed. This step is crucial because without scikit-learn, you won't have access to the machine learning tools needed for the following steps.

# In[ ]:





# ### Step 2: Loading and Exploring the Breast Cancer Dataset
# 
# **Generalized Explanation:**
# ```python
# # Import necessary libraries
# from sklearn.datasets import load_breast_cancer
# import pandas as pd
# 
# # Load the dataset
# breast_cancer = load_breast_cancer()
# 
# # Convert to a DataFrame for easier handling
# data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
# data['target'] = breast_cancer.target
# 
# # Display the first few rows of the dataset
# data.head()
# ```
# 
# **Why It Was Needed:**
# - **Importing Libraries:** 
#   - `from sklearn.datasets import load_breast_cancer`: This imports a function from scikit-learn that loads the breast cancer dataset, which is a commonly used dataset for classification tasks in machine learning.
#   - `import pandas as pd`: Pandas is imported to work with data in a structured manner using DataFrames.
# 
# - **Loading the Dataset:**
#   - `breast_cancer = load_breast_cancer()`: Loads the breast cancer dataset into the `breast_cancer` variable. This dataset contains features and labels for breast cancer diagnosis.
# 
# - **Converting to DataFrame:**
#   - `data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)`: Converts the dataset from scikit-learn's Bunch format to a pandas DataFrame for easier manipulation and analysis. This step assigns feature names as column headers.
#   - `data['target'] = breast_cancer.target`: Adds a column named 'target' to the DataFrame, which contains the labels (0 for malignant, 1 for benign).
# 
# - **Displaying Data:**
#   - `data.head()`: Displays the first few rows of the DataFrame to inspect the structure and content of the dataset.
# 
# **Discussion of Results:**
# - After executing this code, the output will show the first few rows of the breast cancer dataset, including features such as mean radius, mean texture, and so on, along with the corresponding target values (0 or 1).
# - This step is crucial as it allows you to understand the dataset's format, features, and initial data quality. It sets the stage for further preprocessing, exploration, and modeling steps in machine learning tasks.

# In[ ]:





# ### Step 3: Checking the Distribution of Target Labels
# 
# **Generalized Explanation:**
# ```python
# # Check the distribution of target labels
# data['target'].value_counts()
# ```
# 
# **Why It Was Needed:**
# - **Checking Target Label Distribution:**
#   - `data['target'].value_counts()`: This command calculates and displays the count of each unique value in the 'target' column of the DataFrame. In this case, it shows how many instances belong to each class (0 for malignant, 1 for benign) in the breast cancer dataset.
# 
# **Why Did It Show/Discussion of Results:**
# - The output of this command will provide a numerical summary of the distribution of target labels in the dataset. It's important to check this because:
#   - It helps you understand the balance or imbalance between different classes in the dataset. Imbalance can affect model training and evaluation.
#   - For classification tasks, understanding the distribution of classes helps in choosing appropriate evaluation metrics and handling class imbalance if present.
#   
# - For instance, if the output shows that there are significantly more instances of benign tumors (class 1) compared to malignant tumors (class 0), this imbalance might need to be addressed through techniques like oversampling, undersampling, or using appropriate evaluation metrics like precision, recall, and F1-score.
# 
# - Overall, checking the distribution of target labels is a fundamental step in understanding your dataset and preparing it for further analysis and modeling in machine learning.

# In[ ]:





# ### Step 4: Checking for Missing Values
# 
# **Generalized Explanation:**
# ```python
# # Check for missing values
# missing_values = data.isnull().sum()
# print("Missing values in each column:\n", missing_values)
# ```
# 
# **Why It Was Needed:**
# - **Checking for Missing Values:**
#   - `data.isnull().sum()`: This checks each column in the DataFrame (`data`) for missing values (NaN or None) and sums up the counts of missing values for each column.
# 
# **Why Did It Show/Discussion of Results:**
# - The output of this code snippet will display the number of missing values in each column of the dataset.
# - It's crucial to check for missing values because:
#   - Missing data can impact the performance of machine learning models if not handled properly.
#   - Understanding the extent of missing data helps in deciding whether to impute missing values, remove rows or columns with missing values, or use models that can handle missing data.
#   
# - If the output indicates missing values in certain columns, further actions might be required, such as:
#   - Imputation (replacing missing values with a meaningful estimate like mean, median, or mode).
#   - Dropping columns with a high proportion of missing values if they are not critical for analysis or modeling.
#   
# - Ensuring data quality by addressing missing values appropriately is essential for reliable machine learning results.
# 
# This step ensures that the dataset is clean and ready for preprocessing and modeling stages in machine learning tasks.

# In[ ]:





# ### Step 5: Preprocessing and Splitting the Dataset
# 
# **Generalized Explanation:**
# ```python
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# 
# # Separate features and target
# X = data.drop(columns=['target'])
# y = data['target']
# 
# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# 
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# ```
# 
# **Why It Was Needed:**
# - **Importing Libraries:**
#   - `from sklearn.preprocessing import StandardScaler`: Imports the StandardScaler class from scikit-learn, which is used to standardize features by removing the mean and scaling to unit variance.
#   - `from sklearn.model_selection import train_test_split`: Imports the train_test_split function from scikit-learn, which splits datasets into random train and test subsets.
# 
# - **Separating Features and Target:**
#   - `X = data.drop(columns=['target'])`: Extracts the features (independent variables) from the DataFrame `data` by dropping the 'target' column.
#   - `y = data['target']`: Extracts the target variable (dependent variable) from the DataFrame `data`.
# 
# - **Scaling the Features:**
#   - `scaler = StandardScaler()`: Initializes an instance of StandardScaler.
#   - `X_scaled = scaler.fit_transform(X)`: Standardizes the feature data (`X`) by fitting the scaler on `X` and transforming it to have zero mean and unit variance.
# 
# - **Splitting the Dataset:**
#   - `train_test_split(X_scaled, y, test_size=0.2, random_state=42)`: Splits the standardized feature data (`X_scaled`) and the target variable (`y`) into training and testing sets.
#     - `X_train, X_test`: Training and testing feature sets.
#     - `y_train, y_test`: Training and testing target sets.
#     - `test_size=0.2`: Specifies that 20% of the data should be used for testing, and 80% for training.
#     - `random_state=42`: Sets a random seed for reproducibility, ensuring that the split is the same each time the code is run.
# 
# **Discussion of Results:**
# - This step is crucial in preparing the data for machine learning models:
#   - **Standardization**: Scaling the features (`X`) ensures that each feature contributes equally to model training and prevents features with larger scales from dominating the learning process.
#   - **Train-Test Split**: Dividing the data into training and testing sets allows for evaluating the model's performance on unseen data, which helps in assessing its generalization ability.
#   
# - After running this code, you will have:
#   - `X_train` and `y_train`: Training data and labels used to train machine learning models.
#   - `X_test` and `y_test`: Testing data and labels used to evaluate the model's performance.
#   
# - These datasets (`X_train`, `X_test`, `y_train`, `y_test`) are now ready for use in training and evaluating machine learning models such as classifiers (e.g., logistic regression, support vector machines) on the breast cancer dataset.

# In[ ]:





# ### Step 6: Training and Evaluating a Decision Tree Classifier
# 
# **Generalized Explanation:**
# ```python
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# 
# # Initialize the model
# dt_model = DecisionTreeClassifier(random_state=42)
# 
# # Train the model
# dt_model.fit(X_train, y_train)
# 
# # Make predictions
# y_pred_dt = dt_model.predict(X_test)
# 
# # Evaluate the model
# print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
# ```
# 
# **Why It Was Needed:**
# - **Importing Libraries:**
#   - `from sklearn.tree import DecisionTreeClassifier`: Imports the DecisionTreeClassifier class from scikit-learn, which is a classifier based on decision trees.
#   - `from sklearn.metrics import classification_report`: Imports the classification_report function from scikit-learn, which generates a detailed classification report showing metrics like precision, recall, F1-score, and support.
# 
# - **Initializing and Training the Model:**
#   - `dt_model = DecisionTreeClassifier(random_state=42)`: Initializes a decision tree classifier model with `random_state=42` for reproducibility.
#   - `dt_model.fit(X_train, y_train)`: Trains the decision tree classifier using the training data (`X_train` and `y_train`).
# 
# - **Making Predictions:**
#   - `y_pred_dt = dt_model.predict(X_test)`: Uses the trained model to predict the target variable (`y_pred_dt`) for the test data (`X_test`).
# 
# - **Evaluating the Model:**
#   - `classification_report(y_test, y_pred_dt)`: Computes and prints a classification report comparing the predicted labels (`y_pred_dt`) against the actual labels (`y_test`) from the test set.
#     - The classification report includes metrics such as precision, recall, F1-score, and support for each class (0 and 1 in this case).
# 
# **Discussion of Results:**
# - After executing this code:
#   - The classification report will provide insights into how well the decision tree classifier performed on the test data.
#   - Metrics like precision, recall, and F1-score are essential for evaluating the model's performance, especially in the context of a binary classification task (malignant vs. benign breast cancer).
#   
# - Interpretation of the classification report:
#   - **Precision**: Measures the accuracy of positive predictions (e.g., the percentage of predicted benign cases that are actually benign).
#   - **Recall**: Measures the proportion of actual positives that were correctly identified (e.g., the percentage of actual benign cases that were correctly identified as benign).
#   - **F1-score**: Harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.
#   - **Support**: Number of occurrences of each class in `y_test`.
#   
# - By examining these metrics, you can assess whether the decision tree classifier is effective for the task of classifying breast cancer cases based on the provided features. Adjustments to the model or further exploration of feature importance may be warranted based on these results.

# In[ ]:





# ### Step 7: Training and Evaluating a Logistic Regression Classifier
# 
# **Generalized Explanation:**
# ```python
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# 
# # Initialize the model
# lr_model = LogisticRegression(random_state=42)
# 
# # Train the model
# lr_model.fit(X_train, y_train)
# 
# # Make predictions
# y_pred_lr = lr_model.predict(X_test)
# 
# # Evaluate the model
# print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
# ```
# 
# **Why It Was Needed:**
# - **Importing Libraries:**
#   - `from sklearn.linear_model import LogisticRegression`: Imports the LogisticRegression class from scikit-learn, which implements logistic regression for classification tasks.
#   - `from sklearn.metrics import classification_report`: Imports the classification_report function from scikit-learn to generate a detailed classification report.
# 
# - **Initializing and Training the Model:**
#   - `lr_model = LogisticRegression(random_state=42)`: Initializes a logistic regression model with `random_state=42` for reproducibility.
#   - `lr_model.fit(X_train, y_train)`: Trains the logistic regression model using the training data (`X_train` and `y_train`).
# 
# - **Making Predictions:**
#   - `y_pred_lr = lr_model.predict(X_test)`: Uses the trained logistic regression model to predict the target variable (`y_pred_lr`) for the test data (`X_test`).
# 
# - **Evaluating the Model:**
#   - `classification_report(y_test, y_pred_lr)`: Computes and prints a classification report comparing the predicted labels (`y_pred_lr`) against the actual labels (`y_test`) from the test set.
#     - The classification report includes metrics such as precision, recall, F1-score, and support for each class.
# 
# **Discussion of Results:**
# - After running this code:
#   - The classification report will provide insights into how well the logistic regression classifier performed on the test data.
#   - Metrics such as precision, recall, and F1-score are crucial for evaluating the model's performance in the context of a binary classification task (malignant vs. benign breast cancer).
#   
# - Interpretation of the classification report:
#   - **Precision**: Measures the accuracy of positive predictions (e.g., the percentage of predicted benign cases that are actually benign).
#   - **Recall**: Measures the proportion of actual positives that were correctly identified (e.g., the percentage of actual benign cases that were correctly identified as benign).
#   - **F1-score**: Harmonic mean of precision and recall, providing a single metric to evaluate the model's performance.
#   - **Support**: Number of occurrences of each class in `y_test`.
#   
# - Comparing the results with those from the decision tree classifier (Step 6) can help in determining which model performs better for this specific dataset and task. Additionally, examining feature importance or conducting further analysis may reveal insights into the factors driving the predictions of each model.

# In[ ]:





# ### Step 7: Explanation of Decision Tree and Logistic Regression
# 
# #### Decision Tree
# - **Definition:** A Decision Tree is a hierarchical structure where each internal node represents a decision based on a feature attribute, each branch represents the outcome of the decision, and each leaf node represents a class label.
# - **Why to use:** Decision Trees are easy to understand and interpret, making them useful for explaining decisions to non-technical stakeholders. They can handle both numerical and categorical data and can capture non-linear relationships.
# - **When to use:** Use Decision Trees when interpretability is important, and when the data has a mix of feature types or non-linear relationships.
# - **When not to use:** Decision Trees can overfit, especially with deep trees or small datasets. They may not perform well with very large datasets without proper pruning or ensemble methods.
# 
# #### Logistic Regression
# - **Definition:** Logistic Regression is a linear model used for binary classification. It predicts the probability of an instance belonging to a particular class using a logistic function.
# - **Why to use:** Logistic Regression is efficient, interpretable, and works well with linearly separable data. It provides probabilities of class membership.
# - **When to use:** Use Logistic Regression when the relationship between features and target is linear or can be approximated as linear. It is also suitable when you need probabilistic outputs and straightforward interpretation of feature importance.
# - **When not to use:** Logistic Regression may not perform well when there are complex, non-linear relationships between features and target variables. It assumes that the features are independent of each other and there is little multicollinearity.
# 
# ### Analyzing the Results and Recommending a Model
# 
# #### Decision Tree Results:
# - **Precision:** 0.93 for class 0, 0.96 for class 1
# - **Recall:** 0.93 for class 0, 0.96 for class 1
# - **F1-score:** 0.93 for class 0, 0.96 for class 1
# - **Accuracy:** 0.95
# 
# #### Logistic Regression Results:
# - **Precision:** 0.98 for class 0, 0.97 for class 1
# - **Recall:** 0.95 for class 0, 0.99 for class 1
# - **F1-score:** 0.96 for class 0, 0.98 for class 1
# - **Accuracy:** 0.97
# 
# ### Recommendation
# Based on the classification reports:
# 
# 1. **Performance:**
#    - **Logistic Regression** shows higher accuracy (0.97) compared to Decision Tree (0.95).
#    - Logistic Regression also demonstrates higher precision, recall, and F1-score across both classes, indicating superior overall performance.
# 
# 2. **Use Case Consideration:**
#    - If interpretability is crucial and explaining decisions is important, **Decision Tree** might be preferred due to its tree-like structure and ease of understanding.
#    - For maximizing predictive accuracy and handling linear relationships effectively, **Logistic Regression** is recommended.
# 
# 3. **Complexity and Overfitting:**
#    - Decision Trees can overfit the training data, especially with deep trees. Regularization techniques or ensemble methods like Random Forests can mitigate this issue.
#    - Logistic Regression tends to generalize better with simpler assumptions and fewer parameters.
# 
# ### Conclusion:
# **Logistic Regression** is recommended based on the provided results. It exhibits higher accuracy and better performance metrics (precision, recall, F1-score) for both classes, suggesting it will generalize well to new, unseen data. Logistic Regression also provides probabilistic outputs and is less susceptible to overfitting compared to Decision Trees. However, if the interpretability of the model is crucial and you can manage potential overfitting, Decision Trees remain a viable option, particularly when transparency and explainability are paramount.
