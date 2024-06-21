#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:center;color:#00008B">Basics of Python programming <br><br> &nbsp;Scikit-Learn</h1> 
# </header>
# <br><br><br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">ScikitLearn</td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Scikit-learn </h2>
# <ul><li><span style="color:#00008B;font-size:20px">
# Fast and powerful library for predictive data analysis (machine learning in python). <br><br>
# Developed by David Cournapeau. <br><br>
# Built on Numpy, Scipy and Matplotlib package. <br><br>
# This library is focused on modeling the data. <br><br>
# <b>Read More:</b> <a href="https://scikit-learn.org/stable/">Scikit-learn </a>
# </span></li></ul>
#  
# <br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">ScikitLearn</td></tr></table>
# </footer>

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Scikit-learn</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Dataset transformation - preprocessing (feature selection, normalisation), dimensionality reduction. <br><br>
# Supervised and Unsupervised Learning. <br><br>
# Model selection and evaluation. <br><br>
# Visualisation. <br><br>
# Github: <a href="https://github.com/scikit-learn/scikit-learn"> scikit-learn </a> <br><br>
# Read more: <a href="https://scikit-learn.org/stable/user_guide.html"> User Guide </a>
# </span></li>
#     </ul>
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:center;background-color:blue;color:white;font-size:80%;"> Programming in Python </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;">ScikitLearn</td></tr></table>
# </footer>
# 
# </div>

# In[13]:


# Calling the libraries for data loading (pandas) and modeling(scikit-learn: sklearn)

import pandas as p

import sklearn as s

from sklearn import tree as t # Import a specific package from the library

from sklearn.model_selection import train_test_split as tts


# In[ ]:


# To know more about sklearn and functionalities use help

help(s)


# In[14]:


# Importing a dataset from a file
dataset = p.read_csv('/Users/farah/Desktop/iris.csv')


# In[15]:


# Spliting the data into training and test sets: train_test_split as tts
training, test = tts(dataset)


# In[25]:


# Selecting the input features - training set
training_features = training[['Sepal Length(cm)','Sepal Width (cm)','Petal Length (cm)','Petal Width (cm)']]

print(training_features)


# In[26]:


# Selecting the output (response or class label) - training set
training_response = training[['Species']]

print(training_response)


# In[27]:


# Define the input features - test set
test_features = test[['Sepal Length(cm)','Sepal Width (cm)','Petal Length (cm)','Petal Width (cm)']]


# In[28]:


print(test_features)


# In[29]:


# Define the output (response or class label) - test set
test_actual_response = test[['Species']]


# In[30]:


print(test_actual_response)


# In[31]:


# calling a classification algorithm - Decision Tree

model1 = s.tree.DecisionTreeClassifier()

print(model1)


# In[32]:


# apply the algorithm on the dataset

model1.fit(training_features, training_response)


# In[4]:


# Plot the tree

s.tree.plot_tree(model1)


# In[34]:


# Test the model on the test dataset
test_pred_response = model1.predict(test_features)

print(test_pred_response)


# In[35]:


# Model Evaluation - confusion matrix with the performance measures

print(s.metrics.confusion_matrix(test_actual_response, test_pred_response))

print(s.metrics.classification_report(test_actual_response, test_pred_response))


# In[ ]:


# Kfoldcrossvalidation will help us to remove the bias and make the data same
# You will have to run it 5 times

#K shouldn't be the higher than the actual data

# It we are doing it the way we just did now, make sure it is not random sampling

