#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) on the Iris Dataset

# #### Step 1: Data Summarization

# In[1]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


# In[2]:


# Load the Iris dataset
iris = load_iris()
iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_dataset['species'] = iris.target


# In[3]:


# Mapping target values to actual species names
iris_dataset['species'] = iris_dataset['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


# In[4]:


# Descriptive statistics
desc_stats = iris_dataset.describe()
print("Descriptive Statistics:\n", desc_stats)


# In[6]:


# Correlation matrix
corr_matrix = iris_dataset.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr_matrix)


# In[ ]:





# #### Step 2: Data Visualization

# In[7]:


# Import necessary libraries
import matplotlib.pyplot as plt


# In[9]:


# Histograms
iris_dataset.hist(figsize=(10, 8))
plt.suptitle('Histograms of Iris Dataset Features')

# Specify the file path where you want to save the figure
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 2 - EDA on the Iris Dataset/EDA Images/iris_histograms.png'

# Save the figure
plt.savefig(file_path, dpi=300)

plt.show()


# In[ ]:





# #### Step 3: Feature Distribution Analysis Across Species

# In[10]:


# Violin plots to visualize feature distributions across species
plt.figure(figsize=(15, 10))
for i, feature in enumerate(iris_dataset.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=iris_dataset)
    plt.title(f'Violin Plot of {feature} by Species')
plt.tight_layout()

# Specify the file path where you want to save the figure
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 2 - EDA on the Iris Dataset/EDA Images/iris_Violinplot.png'

# Save the figure
plt.savefig(file_path, dpi=300)
plt.show()


# In[ ]:





# #### Step 4: Hypothesis Testing

# In[11]:


# Import necessary library
from scipy.stats import f_oneway

# ANOVA test to determine if there are significant differences between species
features = iris_dataset.columns[:-1]
anova_results = {feature: f_oneway(iris_dataset[iris_dataset['species'] == 'setosa'][feature],
                                   iris_dataset[iris_dataset['species'] == 'versicolor'][feature],
                                   iris_dataset[iris_dataset['species'] == 'virginica'][feature])
                 for feature in features}

# Print ANOVA results
for feature, result in anova_results.items():
    print(f"ANOVA test for {feature}: F-statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4e}")

