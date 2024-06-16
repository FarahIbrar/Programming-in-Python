#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction

# ### Dimensionality Reduction
# 
# **Description**: Apply dimensionality reduction techniques like PCA (Principal Component Analysis) or t-SNE (t-distributed Stochastic Neighbor Embedding) to visualize the Iris dataset in lower dimensions.

# In[ ]:





# ### Detailed Steps:
# 
# 1. **Feature scaling and data transformation**.
# 2. **Applying PCA or t-SNE and visualizing the reduced data**.
# 3. **Interpretability**: Understanding how different species are represented in reduced dimensions.
# 4. **Clustering in reduced space**: Do clusters form that correspond to species?

# In[ ]:





# ### Step-by-Step Implementation:

# #### Step 1: Feature Scaling and Data Transformation
# 
# ```python
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# 
# # Load the Iris dataset
# iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')
# 
# # Separate features and target
# X = iris_dataset.drop(columns='species')
# y = iris_dataset['species']
# 
# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ```

# #### **Explanation**: 
# 
# - **Overall Explanation**: This step involves loading the Iris dataset, separating the features (measurements) from the target variable (species), and standardizing the features to have a mean of 0 and a standard deviation of 1.
# - **Why?**: PCA and t-SNE are sensitive to the scale of the features. Standardizing the data ensures that all features contribute equally to the analysis, preventing any single feature from dominating the results.

# In[ ]:





# #### Step 2: Applying PCA and Visualizing the Reduced Data
# 
# ```python
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import seaborn as sns
# 
# # Apply PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# 
# # Create a DataFrame for PCA results
# pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
# pca_df = pd.concat([pca_df, y], axis=1)
# 
# # Plot PCA results
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='Set1')
# plt.title('PCA of Iris Dataset')
# plt.show()
# ```

# #### **Explanation**: 
# 
# - **Overall Explanation**: This step applies PCA to the standardized data, reducing the dimensionality to two principal components. It then creates a DataFrame for the PCA results and visualizes the data using a scatter plot.
# - **Why?**: PCA reduces the dimensionality of the data by finding new axes (principal components) that maximize variance. By projecting the data onto these axes, we can visualize the dataset in 2D while preserving as much variance as possible, helping us to identify any patterns or clusters in the data.

# In[ ]:





# #### Step 3: Applying t-SNE and Visualizing the Reduced Data
# 
# ```python
# from sklearn.manifold import TSNE
# 
# # Apply t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(X_scaled)
# 
# # Create a DataFrame for t-SNE results
# tsne_df = pd.DataFrame(data=X_tsne, columns=['Dim1', 'Dim2'])
# tsne_df = pd.concat([tsne_df, y], axis=1)
# 
# # Plot t-SNE results
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Dim1', y='Dim2', hue='species', data=tsne_df, palette='Set1')
# plt.title('t-SNE of Iris Dataset')
# plt.show()
# ```

# #### **Explanation**: 
# 
# - **Overall Explanation**: This step applies t-SNE to the standardized data, reducing the dimensionality to two dimensions. It then creates a DataFrame for the t-SNE results and visualizes the data using a scatter plot.
# - **Why?**: t-SNE is a non-linear dimensionality reduction technique that is particularly good at preserving the local structure of the data, making it ideal for visualizing clusters in high-dimensional data. It helps us to see how the data points are related locally and globally.

# In[ ]:





# #### Step 4: Interpretability and Clustering in Reduced Space
# 
# - **PCA Interpretation**:
#   - The scatter plot of the first two principal components shows how the iris species are represented in the reduced space. If distinct clusters form, it suggests that the species are well-separated in the original feature space.
# - **t-SNE Interpretation**:
#   - The scatter plot of the t-SNE dimensions helps visualize the local structure of the data. If the species form distinct clusters, it indicates that t-SNE successfully preserved the local relationships between samples.
# - **Evaluating Clusters**:
#   - Visual inspection of the scatter plots can help identify if natural clusters corresponding to the species are formed.

# In[ ]:





# ### Summary of Results:
# 
# - **PCA Results**:
#   - The PCA plot revealed how the iris species are distributed along the first two principal components. Distinct clusters for each species were observed, indicating good separability in the original feature space.
# - **t-SNE Results**:
#   - The t-SNE plot provided a more detailed view of the local structure, showing well-defined clusters for each species. This confirmed that the iris species have distinct characteristics that can be captured through dimensionality reduction.

# Both PCA and t-SNE demonstrated that the iris species form natural clusters based on their sepal and petal measurements, highlighting the effectiveness of these techniques for visualizing high-dimensional data in a lower-dimensional space.
