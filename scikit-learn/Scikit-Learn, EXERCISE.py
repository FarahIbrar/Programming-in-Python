#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Scikit-Learn, EXERCISE</h1>
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

# In[23]:


get_ipython().system('pip install scikit-learn')


# In[24]:


# Import necessary libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
breast_cancer = load_breast_cancer()

# Convert to a DataFrame for easier handling
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target

# Display the first few rows of the dataset
data.head()


# In[25]:


# Check the distribution of target labels
data['target'].value_counts()


# In[30]:


# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)


# In[27]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop(columns=['target'])
y = data['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Initialize the model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))


# In[32]:


from sklearn.linear_model import LogisticRegression

# Initialize the model
lr_model = LogisticRegression(random_state=42)

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))


# ## Which model is better?
# 
# - Logistic Regression is recommended based on the given results. 
# - It demonstrates higher accuracy and better performance metrics (precision, recall, F1-score) for both classes, which suggests it will generalize better to unseen data. 
# - It also provides the added benefit of probabilistic predictions and is less likely to overfit compared to a Decision Tree.
