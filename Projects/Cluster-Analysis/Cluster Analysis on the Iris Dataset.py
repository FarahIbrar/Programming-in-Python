#!/usr/bin/env python
# coding: utf-8

# # Cluster Analysis on the Iris Dataset

# #### Step 1: Feature Selection and Preprocessing

# In[3]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load the Iris dataset
iris = load_iris()
iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_dataset['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[4]:


# Selecting features (all numeric features)
features = iris_dataset.iloc[:, :-1]


# In[5]:


# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[6]:


# Applying PCA to reduce dimensions (optional for better visualization)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)


# In[ ]:





# #### Step 2: Choosing the Number of Clusters

# In[7]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[9]:


# Using the Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicitly setting n_init to suppress the warning
    kmeans.fit(features_pca)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.tight_layout()

# Specify the file path to save the plot
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/CA Images/elbow_method_plot.png'  # Replace with your desired file path

# Save the plot to a file
plt.savefig(file_path)

# Show the plot
plt.show()


# In[ ]:





# #### Step 3: Applying K-means Clustering

# In[11]:


# Applying K-means with the optimal number of clusters (assume 3 from Elbow method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init to suppress the warning
clusters = kmeans.fit_predict(features_pca)

# Adding the cluster labels to the dataset
iris_dataset['Cluster'] = clusters

# Save the updated dataset with cluster labels to a file
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/File/iris_dataset_with_clusters.csv'  # Replace with your desired file path
iris_dataset.to_csv(file_path, index=False)

# Print the first few rows of the updated dataset
print(iris_dataset.head())


# In[13]:


# Display the cluster centers in the PCA reduced space
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Cluster Centers')
plt.title('K-means Clustering with 3 clusters on Iris Dataset (PCA-reduced Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Specify the file path to save the plot
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/CA Images/Kmeans_clustring_plot.png'  # Replace with your desired file path

plt.show()


# In[ ]:





# #### Step 4: Evaluating Cluster Quality

# In[14]:


from sklearn.metrics import silhouette_score

# Calculating the Silhouette Score
silhouette_avg = silhouette_score(features_pca, clusters)
print(f'Silhouette Score: {silhouette_avg}')


# In[15]:


import seaborn as sns

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=iris_dataset['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('K-means Clustering on Iris Dataset (PCA-reduced Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')

# Specify the file path to save the plot
file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/CA Images/Kmeans_clustering_plot.png'  # Replace with your desired file path

plt.show()

