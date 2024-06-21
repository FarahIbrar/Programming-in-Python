#!/usr/bin/env python
# coding: utf-8

# # Scikit-Learn - Notes

# ## Step-by-Step Explanation

# ### Importing Libraries
# 
# ```python
# import pandas as p
# import sklearn as s
# from sklearn import tree as t
# from sklearn.model_selection import train_test_split as tts
# ```
# 
# #### **What does this step do?**
# - Imports the necessary libraries for data handling and machine learning.
# - `pandas` (aliased as `p`) is used for data manipulation and analysis.
# - `sklearn` (aliased as `s`) is a machine learning library.
# - Specifically imports the `tree` module from `sklearn` as `t` for decision tree modeling.
# - Imports `train_test_split` from `sklearn.model_selection` as `tts` to split data into training and testing sets.
# 
# #### **Why is this needed?**
# - To utilize `pandas` for reading and handling datasets.
# - To access machine learning algorithms and tools provided by `sklearn`.
# 
# #### **What does it show?**
# - It sets up the environment by importing all the necessary tools for the tasks ahead.

# In[ ]:





# ### Getting Help on sklearn
# 
# ```python
# help(s)
# ```
# 
# #### **What does this step do?**
# - Displays the help documentation for the `sklearn` library.
# 
# #### **Why is this needed?**
# - To provide an overview of available functionalities within `sklearn`.
# 
# #### **What does it show?**
# - Detailed documentation and descriptions of the `sklearn` library's modules and functions.

# In[ ]:





# ### Loading the Dataset
# 
# ```python
# dataset = p.read_csv('/Users/farah/Desktop/iris.csv')
# ```
# 
# #### **What does this step do?**
# - Reads the dataset from a CSV file into a `pandas` DataFrame.
# 
# #### **Why is this needed?**
# - To load the data into a format that can be easily manipulated and analyzed.
# 
# #### **What does it show?**
# - The dataset is now available in a DataFrame for further processing.

# In[ ]:





# ### Splitting the Data
# 
# ```python
# training, test = tts(dataset)
# ```
# 
# #### **What does this step do?**
# - Splits the dataset into training and testing sets using `train_test_split`.
# 
# #### **Why is this needed?**
# - To create separate sets for training the model and testing its performance.
# 
# #### **What does it show?**
# - Two new DataFrames: `training` and `test`, which are subsets of the original dataset.

# In[ ]:





# ### Selecting Training Features
# 
# ```python
# training_features = training[['Sepal Length(cm)','Sepal Width (cm)','Petal Length (cm)','Petal Width (cm)']]
# print(training_features)
# ```
# 
# #### **What does this step do?**
# - Selects the feature columns (input variables) from the training set.
# 
# #### **Why is this needed?**
# - To isolate the input features needed for training the model.
# 
# #### **What does it show?**
# - The feature columns of the training set, printed to verify the selection

# In[ ]:





# ### Selecting Training Response
# 
# ```python
# training_response = training[['Species']]
# print(training_response)
# ```
# 
# #### **What does this step do?**
# - Selects the target column (output variable) from the training set.
# 
# #### **Why is this needed?**
# - To isolate the target variable that the model will learn to predict.
# 
# #### **What does it show?**
# - The target column of the training set, printed to verify the selection.

# In[ ]:





# ### Selecting Test Features
# 
# ```python
# test_features = test[['Sepal Length(cm)','Sepal Width (cm)','Petal Length (cm)','Petal Width (cm)']]
# print(test_features)
# ```
# 
# #### **What does this step do?**
# - Selects the feature columns (input variables) from the test set.
# 
# #### **Why is this needed?**
# - To isolate the input features needed for testing the model.
# 
# #### **What does it show?**
# - The feature columns of the test set, printed to verify the selection.

# In[ ]:





# ### Define the Output (Response or Class Label) - Test Set
# 
# ```python
# test_actual_response = test[['Species']]
# print(test_actual_response)
# ```
# 
# #### **What does this step do?**
# - Selects the target column (output variable) from the test set.
# 
# #### **Why is this needed?**
# - To isolate the actual target variable values for the test set.
# 
# #### **What does it show?**
# - The target column of the test set, printed to verify the selection.

# In[ ]:





# ### Calling a Classification Algorithm - Decision Tree
# 
# ```python
# model1 = s.tree.DecisionTreeClassifier()
# print(model1)
# ```
# 
# #### **What does this step do?**
# - Creates an instance of the DecisionTreeClassifier from `sklearn`.
# 
# #### **Why is this needed?**
# - To define the model that will be used for classification.
# 
# #### **What does it show?**
# - The parameters and default settings of the Decision Tree Classifier.

# In[ ]:





# ### Apply the Algorithm on the Dataset
# 
# ```python
# model1.fit(training_features, training_response)
# ```
# 
# #### **What does this step do?**
# - Trains the Decision Tree model using the training data.
# 
# #### **Why is this needed?**
# - To teach the model to associate features with the corresponding target classes.
# 
# #### **What does it show?**
# - The trained Decision Tree model.

# In[ ]:





# ### Plot the Tree
# 
# ```python
# s.tree.plot_tree(model1)
# ```
# 
# #### **What does this step do?**
# - Visualizes the trained Decision Tree model.
# 
# #### **Why is this needed?**
# - To understand the structure and decisions made by the model.
# 
# #### **What does it show?**
# - A graphical representation of the Decision Tree with decision nodes and leaf nodes.

# In[ ]:





# ### Test the Model on the Test Dataset
# 
# ```python
# test_pred_response = model1.predict(test_features)
# print(test_pred_response)
# ```
# 
# #### **What does this step do?**
# - Uses the trained model to predict the target variable for the test features.
# 
# #### **Why is this needed?**
# - To evaluate how well the model generalizes to new, unseen data.
# 
# #### **What does it show?**
# - The predicted target values for the test set.

# In[ ]:





# ### Model Evaluation - Confusion Matrix and Classification Report
# 
# ```python
# print(s.metrics.confusion_matrix(test_actual_response, test_pred_response))
# print(s.metrics.classification_report(test_actual_response, test_pred_response))
# ```
# 
# #### **What does this step do?**
# - Computes and prints the confusion matrix and classification report.
# 
# #### **Why is this needed?**
# - To evaluate the performance of the model on the test set.
# 
# #### **What does it show?**
# - The confusion matrix showing the counts of true positive, true negative, false positive, and false negative predictions.
# - The classification report showing precision, recall, F1-score, and support for each class, as well as the overall accuracy and other metrics.

# In[ ]:





# #### K-Fold Cross Validation Notes only
# 
# ```python
# # K-fold cross-validation will help us to remove the bias and make the data same
# # You will have to run it 5 times
# # K shouldn't be the higher than the actual data
# # If we are doing it the way we just did now, make sure it is not random sampling
# ```
# 
# **What does this step do?**
# - Comments on the intention to use K-fold cross-validation for model evaluation.
# 
# **Why is this needed?**
# - K-fold cross-validation provides a more reliable estimate of model performance by using multiple train-test splits.
# 
# **What does it show?**
# - Describes the need for K-fold cross-validation and suggests running it 5 times, ensuring the value of K is reasonable.
