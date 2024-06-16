#!/usr/bin/env python
# coding: utf-8

#  # Dimensionality Reduction

# #### Step 1: Feature Scaling and Data Transformation

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Separate features and target
X = iris_dataset.drop(columns='species')
y = iris_dataset['species']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:





# #### Step 2: Applying PCA and Visualizing the Reduced Data

# In[4]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df = pd.concat([pca_df, y], axis=1)

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='Set1')
plt.title('PCA of Iris Dataset')

# Specify the file path to save the plot
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 5 - Dimensionality Reduction/DR Images/PCA_iris_dataset.png'  # Replace with your desired file path

# Save the plot to a file
plt.savefig(file_path)


plt.show()


# In[ ]:





# #### Step 3: Applying t-SNE and Visualizing the Reduced Data

# In[5]:


from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame for t-SNE results
tsne_df = pd.DataFrame(data=X_tsne, columns=['Dim1', 'Dim2'])
tsne_df = pd.concat([tsne_df, y], axis=1)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Dim1', y='Dim2', hue='species', data=tsne_df, palette='Set1')
plt.title('t-SNE of Iris Dataset')

# Specify the file path to save the plot
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 5 - Dimensionality Reduction/DR Images/tSNE_iris_dataset.png'  # Replace with your desired file path

# Save the plot to a file
plt.savefig(file_path)


plt.show()

