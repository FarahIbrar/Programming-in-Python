#!/usr/bin/env python
# coding: utf-8

# # Non Visual Analysis

# In[3]:


import pandas as pd

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Display the first few rows to verify
print(iris_dataset.head())


# ## Missing Value Analysis
# Check for missing values in the dataset.

# In[4]:


# Check for missing values
missing_values = iris_dataset.isnull().sum()
print(missing_values)


# ### Explanation:
# This analysis helps identify any missing values in the dataset, which is a critical step before performing further analysis.

# ## Unique Value Counts
# Count the number of unique values in each column.

# In[5]:


# Count unique values in each column
unique_counts = iris_dataset.nunique()
print(unique_counts)


# ### Explanation:
# Unique value counts give an overview of the categorical diversity and variability in each feature.

# ## Species Distribution
# Calculate the distribution of each species in the dataset.

# In[6]:


# Distribution of species
species_distribution = iris_dataset['species'].value_counts()
print(species_distribution)


# ### Explanation:
# Understanding the distribution of species helps in balancing datasets and analyzing the prevalence of each category.

# ## Skewness and Kurtosis
# Calculate skewness and kurtosis for each feature.

# In[7]:


# Calculate skewness and kurtosis
skewness = iris_dataset.skew()
kurtosis = iris_dataset.kurt()
print(f'Skewness:\n{skewness}\n')
print(f'Kurtosis:\n{kurtosis}\n')


# In[ ]:


To avoid the warning above ^ you can use this code:


# In[24]:


import pandas as pd

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Calculate skewness and kurtosis with numeric_only parameter
skewness = iris_dataset.skew(numeric_only=True)
kurtosis = iris_dataset.kurt(numeric_only=True)

print(f'Skewness:\n{skewness}\n')
print(f'Kurtosis:\n{kurtosis}\n')


# ### Explanation:
# Skewness measures the asymmetry of the distribution of values, while kurtosis measures the "tailedness" of the distribution.

# ## Normality Test
# Perform a normality test (Shapiro-Wilk test) on each feature.

# In[8]:


from scipy.stats import shapiro

# Perform Shapiro-Wilk test for normality
for column in iris_dataset.columns[:-1]:
    stat, p = shapiro(iris_dataset[column])
    print(f'{column} - Statistics={stat}, p={p}')


# ### Explanation:
# The Shapiro-Wilk test checks if a sample comes from a normal distribution. A p-value less than 0.05 indicates non-normality.

# ## Feature Scaling
# Scale the features using StandardScaler.

# In[9]:


from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(iris_dataset.drop('species', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=iris_dataset.columns[:-1])
print(scaled_df.head())


# ### Explanation:
# Feature scaling is essential for many machine learning algorithms, ensuring that each feature contributes equally to the model.

# ## Feature Engineering: Polynomial Features
# Create polynomial features to increase model complexity.

# In[10]:


from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(iris_dataset.drop('species', axis=1))
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(iris_dataset.columns[:-1]))
print(poly_df.head())


# ### Explanation:
# Polynomial features can capture non-linear relationships between variables, potentially improving model performance.

# ## Principal Component Analysis (PCA)
# Reduce dimensionality using PCA and explain variance.

# In[11]:


from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(iris_dataset.drop('species', axis=1))
explained_variance = pca.explained_variance_ratio_
print(f'Explained Variance Ratio: {explained_variance}')


# ### Explanation:
# PCA reduces the dimensionality of the data, capturing the most variance with the fewest components, simplifying further analysis.

# ## Logistic Regression
# Build a logistic regression model to classify species.

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.drop('species', axis=1), iris_dataset['species'], test_size=0.2, random_state=42)

# Build and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy}')


# ### Explanation:
# Logistic regression is a simple yet effective classification algorithm, providing a baseline for comparison with more complex models.

# ## K-Nearest Neighbors (KNN)
# Build and evaluate a KNN classifier.

# In[13]:


from sklearn.neighbors import KNeighborsClassifier

# Build and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy: {accuracy}')


# ### Explanation:
# KNN is a simple, instance-based learning algorithm that classifies based on the majority class of the nearest neighbors.

# ## Decision Tree Classifier
# Build and evaluate a decision tree classifier.

# In[14]:


from sklearn.tree import DecisionTreeClassifier

# Build and train the model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy}')


# ### Explanation:
# Decision trees partition the data into subsets based on feature values, creating a model that is easy to interpret.

# ## Random Forest Classifier
# Build and evaluate a random forest classifier.

# In[15]:


from sklearn.ensemble import RandomForestClassifier

# Build and train the model
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)

# Predict and evaluate
y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy}')


# ### Explanation:
# Random forests combine multiple decision trees to improve generalization and reduce overfitting, providing robust performance.

# ## Support Vector Machine (SVM)
# Build and evaluate an SVM classifier.

# In[16]:


from sklearn.svm import SVC

# Build and train the model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy}')


# ### Explanation:
# SVMs find the optimal hyperplane that separates the data into classes, effective for both linear and non-linear classification.

# ## Cross-Validation
# Perform cross-validation to evaluate model performance.

# In[17]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, iris_dataset.drop('species', axis=1), iris_dataset['species'], cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')


# ### Explanation:
# Cross-validation provides a more reliable estimate of model performance by splitting the data into multiple train-test splits.

# ## Hyperparameter Tuning: Grid Search
# Perform grid search for hyperparameter tuning.

# In[18]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Perform grid search
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Print best parameters
print(f'Best Parameters: {grid.best_params_}')


# ### Explanation:
# Grid search exhaustively searches for the best hyperparameters from the specified parameter grid, optimizing model performance.

# ## Confusion Matrix
# Generate a confusion matrix to evaluate classification performance.

# In[25]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.savefig('/Users/farah/Desktop/confusion_matrix.png')
plt.show()


# ### Explanation:
# A confusion matrix shows the true vs. predicted classifications, helping to evaluate model performance beyond accuracy.

# ## Classification Report
# Generate a classification report with precision, recall, and F1-score.

# In[20]:


from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred)
print(report)


# ### Explanation:
# The classification report provides detailed performance metrics for each class, including precision, recall, and F1-score.

# ## Feature Importance
# Calculate and display feature importance from a tree-based model.

# In[21]:


importances = forest.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': iris_dataset.columns[:-1], 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))


# ### Explanation:
# Feature importance scores indicate the contribution of each feature to the model, helping to identify the most influential features.

# ## ROC Curve
# Plot the Receiver Operating Characteristic (ROC) curve for the model.

# ### Explanation:
# 
# 1. **Importing Required Libraries**: Make sure to import all necessary libraries at the beginning of your code.
# 2. **Binarizing the Output**: The `label_binarize` function converts the multi-class labels into a binary format which is required for ROC curve calculation.
# 3. **Training a Multi-class Model**: `OneVsRestClassifier` is used with an SVM classifier (`SVC`) to train a multi-class model.
# 4. **Computing ROC Curve and AUC**: The ROC curve and the AUC are computed for each class.
# 5. **Plotting ROC Curve**: The ROC curves for each class are plotted, and the plot is saved as a PNG file.
# 
# ### Running the Code:
# 1. Ensure that `y_test`, `y_train`, `X_train`, and `X_test` are defined in your notebook.
# 2. Copy and paste the updated code into a new cell in your Jupyter Notebook.
# 3. Run the cell by pressing `Shift + Enter`.

# In[26]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from itertools import cycle

# Assuming y_test and y_train are defined, and X_train and X_test contain the training and testing features respectively

# Binarize the output
y_test_bin = label_binarize(y_test, classes=iris_dataset['species'].unique())
n_classes = y_test_bin.shape[1]

# Train a multi-class model
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('/Users/farah/Desktop/roc_curve.png')
plt.show()


# ### Explanation:
# The ROC curve illustrates the true positive rate vs. false positive rate, providing insights into the trade-offs between sensitivity and specificity.

# ## Clustering: K-Means
# Perform K-Means clustering on the dataset.

# In[23]:


from sklearn.cluster import KMeans

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(iris_dataset.drop('species', axis=1))

# Add cluster labels to the dataset
iris_dataset['Cluster'] = clusters
print(iris_dataset.head())


# The warning you might see is a future warning that informs you about a default parameter change in the upcoming version of scikit-learn. To avoid this warning, you should explicitly set the `n_init` parameter when creating the `KMeans` object. Therefore the code down below avoids it.
# 
# ### Explanation:
# 
# 1. **Loading the Dataset**: The dataset (`iris.csv`) is loaded into a pandas DataFrame.
# 2. **Performing K-Means Clustering**:
#    - `KMeans` is instantiated with `n_clusters=3` to specify the number of clusters.
#    - `random_state=42` is set for reproducibility.
#    - `n_init=10` is explicitly set to suppress the warning about the default value change.
# 3. **Adding Cluster Labels**: The cluster labels obtained from `fit_predict` are added as a new column (`Cluster`) to the `iris_dataset` DataFrame.
# 4. **Printing the Head of the Dataset**: Displays the first few rows of the dataset with the newly added `Cluster` column.
# 
# ### Running the Code:
# 
# 1. Make sure the path to your `iris.csv` file is correct (`'/Users/farah/Desktop/iris.csv'`).
# 2. Copy and paste the updated code into a new cell in your Jupyter Notebook.
# 3. Run the cell by pressing `Shift + Enter`.

# In[27]:


import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init
clusters = kmeans.fit_predict(iris_dataset.drop('species', axis=1))

# Add cluster labels to the dataset
iris_dataset['Cluster'] = clusters
print(iris_dataset.head())


# ### Explanation:
# K-Means clustering partitions the data into k clusters, identifying natural groupings within the dataset.

# These analyses cover a broad range of techniques, from basic data manipulation to advanced machine learning and statistical analysis. Each task showcases a different aspect of Python's capabilities for data science, providing a comprehensive toolkit for meaningful data analysis.
