#!/usr/bin/env python
# coding: utf-8

# # Different types of plots

# In[1]:


# Import seaborn as sns and matplotlib.pyplot as plt

import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')


# ## Summary Statistics 
# Calculate and display summary statistics for each feature in the dataset.

# In[3]:


# Summary statistics for each feature
summary_stats = iris_dataset.describe()
print(summary_stats)


# ### Explanation:
# This will provide a statistical summary including count, mean, standard deviation, min, and max values for each feature.

# ## Pairwise Relationships (Pairplot)
# Visualize the pairwise relationships between all features.

# In[4]:


sns.pairplot(iris_dataset, hue='species')
plt.savefig('/Users/farah/Desktop/pairplot.png')
plt.show()


# ### Explanation:
# A pairplot shows scatter plots between each pair of features and the distribution of each feature on the diagonal. The `hue parameter` colours the points by species.

# ## Box Plot
# Compare the distribution of each feature across different species.

# In[5]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_dataset, orient="h", palette="Set2")
plt.savefig('/Users/farah/Desktop/boxplot.png')
plt.show()


# ### Explanation:
# A box plot provides a summary of the distribution, central value, and variability of data for each feature, highlighting potential outliers.

# ## Violin Plot
# Similar to a box plot but also shows the kernel density estimation.

# In[6]:


plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="sepal_length", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/violin_sepal_length.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="sepal_width", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/violin_sepal_width.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="petal_length", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/violin_petal_length.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="species", y="petal_width", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/violin_petal_width.png')
plt.show()


# ### Explanation:
# Violin plots combine the features of a box plot and a density plot, providing more information about the distribution of the data.

# ## Swarm Plot
# Visualize the distribution of features with non-overlapping points.

# In[7]:


plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="sepal_length", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/swarm_sepal_length.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="sepal_width", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/swarm_sepal_width.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="petal_length", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/swarm_petal_length.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.swarmplot(x="species", y="petal_width", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/swarm_petal_width.png')
plt.show()


# ### Explanation:
# Swarm plots display points that are adjusted (swarmed) to avoid overlap, giving a clear view of the distribution of the data points.

# ## Joint Plot
# Visualize individual feature relationships with bivariate scatter plots and univariate histograms.

# In[8]:


sns.jointplot(x="sepal_length", y="sepal_width", data=iris_dataset, hue="species")
plt.savefig('/Users/farah/Desktop/joint_sepal.png')
plt.show()

sns.jointplot(x="petal_length", y="petal_width", data=iris_dataset, hue="species")
plt.savefig('/Users/farah/Desktop/joint_petal.png')
plt.show()


# ### Explanation:
# Joint plots combine scatter plots and histograms for two variables, showing their relationship and individual distributions.

# ## Pairwise KDE Plot
# Kernel Density Estimation for pairwise relationships.

# In[9]:


sns.pairplot(iris_dataset, kind="kde", hue="species")
plt.savefig('/Users/farah/Desktop/pairwise_kde.png')
plt.show()


# ### Explanation:
# KDE plots estimate the probability density function of a continuous random variable, providing a smooth curve to understand the distribution.

# ## FacetGrid: Sepal Length
# Facet grid to show distributions of sepal length for each species.

# In[11]:


plot = sns.FacetGrid(iris_dataset, hue="species", height=5)
plot.map(sns.histplot, "sepal_length").add_legend()
plt.savefig('/Users/farah/Desktop/facet_sepal_length.png')
plt.show()


# ### Explanation:
# FacetGrid allows you to create multi-plot grids based on a feature. This example shows histograms of sepal length separated by species.

# ## Boxen Plot
# Enhanced box plot for large datasets.

# In[12]:


plt.figure(figsize=(10, 6))
sns.boxenplot(x="species", y="sepal_length", data=iris_dataset)
plt.savefig('/Users/farah/Desktop/boxen_sepal_length.png')
plt.show()


# ### Explanation:
# Boxen plots provide additional granularity compared to standard box plots, useful for larger datasets.

# ## ECDF Plot
# Empirical Cumulative Distribution Function for features.

# In[19]:


sns.ecdfplot(data=iris_dataset, x="sepal_length", hue="species")
plt.savefig('/Users/farah/Desktop/ecdf_sepal_length.png')
plt.show()

sns.ecdfplot(data=iris_dataset, x="sepal_width", hue="species")
plt.savefig('/Users/farah/Desktop/ecdf_sepal_width.png')
plt.show()

sns.ecdfplot(data=iris_dataset, x="petal_length", hue="species")
plt.savefig('/Users/farah/Desktop/ecdf_petal_length.png')
plt.show()

sns.ecdfplot(data=iris_dataset, x="petal_width", hue="species")
plt.savefig('/Users/farah/Desktop/ecdf_petal_width.png')
plt.show()


# ### Explanation:
# ECDF plots show the proportion of data points less than or equal to each value, giving a cumulative view of the distribution.

# ## Rug Plot
# Show individual data points along with a density plot.

# In[13]:


sns.kdeplot(data=iris_dataset, x="sepal_length", hue="species", fill=True)
sns.rugplot(data=iris_dataset, x="sepal_length", hue="species")
plt.savefig('/Users/farah/Desktop/rug_sepal_length.png')
plt.show()


# ### Explanation:
# Rug plots add small vertical lines to show the distribution of individual data points.

# ## Scatter Plot Matrix
# Matrix of scatter plots for all pairs of features.

# In[14]:


pd.plotting.scatter_matrix(iris_dataset, figsize=(12, 12), diagonal='kde')
plt.savefig('/Users/farah/Desktop/scatter_matrix.png')
plt.show()


# ### Explanation:
# Scatter plot matrices display scatter plots for every pair of features, helping to visualize relationships in the data.

# ## Andrews Curves
# Visual representation of multivariate data.

# In[15]:


from pandas.plotting import andrews_curves
andrews_curves(iris_dataset, "species")
plt.savefig('/Users/farah/Desktop/andrews_curves.png')
plt.show()


# ### Explanation:
# Andrews curves transform the multivariate data into a series of continuous curves, useful for visualizing structure.

# ## Parallel Coordinates
# Visualize multi-dimensional data by plotting each feature on a separate parallel axis.

# In[16]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(iris_dataset, "species")
plt.savefig('/Users/farah/Desktop/parallel_coordinates.png')
plt.show()


# ### Explanation:
# Parallel coordinates plot helps visualize multidimensional categorical data.

# ## RadViz
# Project multi-dimensional data into 2D for visualization.

# In[17]:


from pandas.plotting import radviz
radviz(iris_dataset, "species")
plt.savefig('/Users/farah/Desktop/radviz.png')
plt.show()


# ### Explanation:
# RadViz is a projection method that places each feature on a circle and represents each observation as a point inside the circle.

# ## PCA Biplot
# Perform Principal Component Analysis and plot the first two principal components.

# In[18]:


from sklearn.decomposition import PCA

# Standardize the features
features = iris_dataset.drop('species', axis=1)
features_standardized = (features - features.mean()) / features.std()

# PCA transformation
pca = PCA(n_components=2)
pca_components = pca.fit_transform(features_standardized)

# Create a DataFrame with the PCA components and species
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, iris_dataset[['species']]], axis=1)

# Plot the PCA components
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df)
plt.title('PCA Biplot of Iris Dataset')
plt.savefig('/Users/farah/Desktop/pca_biplot.png')
plt.show()


# ### Explanation:
# Principal Component Analysis (PCA) reduces the dimensionality of the data, projecting it onto the first two principal components, making it easier to visualize the structure.

# By running these analyses, you will get a comprehensive understanding of the Iris dataset from various perspectives. Each plot and visualization provides unique insights into the relationships and distributions of the features within the dataset.
