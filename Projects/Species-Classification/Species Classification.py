#!/usr/bin/env python
# coding: utf-8

# # Project 1: Species Classification

# #### Step 1: Data Preprocessing

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


# In[2]:


# Load the Iris dataset
iris = load_iris()
iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_dataset['species'] = iris.target


# In[3]:


# Display the first few rows of the dataset
print(iris_dataset.head())


# In[8]:


# Check for missing values
print(iris_dataset.isnull().sum())

# There are no missing values in the Iris dataset, so no further action is needed.


# In[4]:


# Separate features (X) and target (y)
X = iris_dataset.drop('species', axis=1)
y = iris_dataset['species']


# In[5]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


# Display the scaled features
print("Scaled features:")
print(pd.DataFrame(X_train_scaled, columns=X_train.columns).head())


# In[ ]:





# #### Step 2: Feature Engineering

# In[10]:


# Creating a new feature for petal area
iris_dataset['petal_area'] = iris_dataset['petal length (cm)'] * iris_dataset['petal width (cm)']

# Log transformation of sepal length
import numpy as np
iris_dataset['sepal length (cm)_log'] = np.log(iris_dataset['sepal length (cm)'])

# Display the dataset with the new and transformed features
print(iris_dataset.head())


# In[ ]:





# #### Step 3: Model Selection and Training 

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting the dataset into features and target variable
X = iris_dataset.drop(columns=['species'])
y = iris_dataset['species']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Training and evaluating each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")


# In[ ]:





# #### Step 4: Model Evaluation and Selection 

# In[12]:


from sklearn.metrics import classification_report, confusion_matrix

# Evaluate each model with detailed metrics
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    
    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")


# In[ ]:





# #### Step 5: Hyperparameter Tuning

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Define hyperparameters for tuning
param_grid = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt'],  # Changed from 'auto' to 'sqrt'
        'max_depth': [None, 10, 20, 30]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
}

# Perform Grid Search CV for each model
best_models = {}
for name, model in models.items():
    print(f"Tuning hyperparameters for {name}")
    grid = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=5, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)
    
    best_models[name] = grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best cross-validated accuracy for {name}: {grid.best_score_}")
    print("\n")


# In[ ]:





# #### Step 6: Evaluate Tuned Models on the Test Set

# In[16]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate each tuned model on the test set
for name, model in best_models.items():
    print(f"Evaluating {name} on the test set")
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.4f}")
    
    # Classification report
    print(f"Classification report for {name}:\n{classification_report(y_test, y_pred)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix for {name}:\n{cm}\n")


# In[ ]:





# #### Step 7: Interpret Results and Draw Conclusions

# In[17]:


# Summarize the results of the best models
for name, model in best_models.items():
    print(f"Final evaluation results for {name}:")

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

# Overall best model based on accuracy
best_model_name = max(best_models, key=lambda name: accuracy_score(y_test, best_models[name].predict(X_test_scaled)))
print(f"The overall best model is: {best_model_name}")

