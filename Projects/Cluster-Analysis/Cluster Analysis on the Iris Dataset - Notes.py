#!/usr/bin/env python
# coding: utf-8

# # Cluster Analysis on the Iris Dataset

# #### Step 1: Feature Selection and Preprocessing
# 
# ``` python
# # Import necessary libraries
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# 
# # Load the Iris dataset
# iris = load_iris()
# iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_dataset['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# 
# # Selecting features (all numeric features)
# features = iris_dataset.iloc[:, :-1]
# 
# # Standardizing the features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)
# 
# # Applying PCA to reduce dimensions (optional for better visualization)
# pca = PCA(n_components=2)
# features_pca = pca.fit_transform(features_scaled)
# ```

# #### **Explanation:**
# 
# **Loading the Iris Dataset:**
# - Load the Iris dataset using `load_iris()` from `sklearn.datasets`.
# - Create a DataFrame with feature names and add a column for species using `pd.Categorical.from_codes()` to map target values to actual species names.
# 
# **Why This Step is Needed:**
# - This step prepares the dataset by loading the Iris data, standardizing the numerical features, and reducing dimensionality using PCA for better visualization of clusters.

# In[ ]:





# #### Step 2: Choosing the Number of Clusters
# 
# ```python
# # Using the Elbow method to find the optimal number of clusters
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicitly setting n_init to suppress the warning
#     kmeans.fit(features_pca)
#     wcss.append(kmeans.inertia_)
# 
# # Plotting the Elbow Method
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS (Within-cluster Sum of Squares)')
# plt.grid(True)
# plt.tight_layout()
# 
# # Specify the file path to save the plot
# file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/CA Images/elbow_method_plot.png'  # Replace with your desired file path
# 
# # Save the plot to a file
# plt.savefig(file_path)
# 
# # Show the plot
# plt.show()
# ```

# #### **Explanation:**
# 
# **Elbow Method for Optimal Clusters:**
# - Use the Elbow method to determine the optimal number of clusters (`n_clusters`) by calculating the Within-cluster Sum of Squares (WCSS) for different numbers of clusters.
# - Plot the WCSS against the number of clusters to identify the elbow point where the rate of decrease sharply slows.
# 
# **Why This Step is Needed:**
# - This step helps in selecting the appropriate number of clusters to use in K-means clustering, providing a balance between cluster compactness and complexity.

# In[ ]:





# #### Step 3: Applying K-means Clustering
# 
# ```python
# # Applying K-means with the optimal number of clusters (assume 3 from Elbow method)
# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init to suppress the warning
# clusters = kmeans.fit_predict(features_pca)
# 
# # Adding the cluster labels to the dataset
# iris_dataset['Cluster'] = clusters
# 
# # Save the updated dataset with cluster labels to a file
# file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/File/iris_dataset_with_clusters.csv'  # Replace with your desired file path
# iris_dataset.to_csv(file_path, index=False)
# 
# # Print the first few rows of the updated dataset
# print(iris_dataset.head())
# ```

# #### **Explanation:**
# 
# **Applying K-means Clustering:**
# - Apply K-means clustering using the optimal number of clusters identified from the Elbow method.
# - Assign cluster labels (`Cluster`) to the original dataset and save the updated dataset with cluster labels to a CSV file.
# 
# **Why This Step is Needed:**
# - This step performs the actual clustering and assigns cluster labels to each data point, providing a way to analyze and understand the data in terms of these clusters.

# 

# #### Step 4: Evaluating Cluster Quality
# 
# ```python
# # Calculating the Silhouette Score
# silhouette_avg = silhouette_score(features_pca, clusters)
# print(f'Silhouette Score: {silhouette_avg}')
# 
# # Visualizing the clusters
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=iris_dataset['Cluster'], palette='viridis', s=100, alpha=0.7)
# plt.title('K-means Clustering on Iris Dataset (PCA-reduced Data)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title='Cluster')
# 
# # Specify the file path to save the plot
# file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 3 - Cluster Analysis on the Iris Dataset/CA Images/Kmeans_clustering_plot.png'  # Replace with your desired file path
# 
# plt.show()
# ```

# #### **Explanation:**
# 
# **Evaluating Cluster Quality:**
# - Calculate the Silhouette Score to evaluate the quality of the clusters.
# - Visualize the clusters in the reduced PCA space to understand the distribution of data points among clusters.
# 
# **Why This Step is Needed:**
# - This step assesses how well-defined the clusters are using the Silhouette score and provides a visual representation of the clustering results.

# In[ ]:





# #### Results Summary
# 
# - **Step 1:** Prepare the Iris dataset by loading it, standardizing features, and reducing dimensionality using PCA.
# - **Step 2:** Determine the optimal number of clusters using the Elbow method.
# - **Step 3:** Apply K-means clustering with the identified optimal number of clusters and add cluster labels to the dataset.
# - **Step 4:** Evaluate the quality of clusters using the Silhouette score and visualize the clusters.
# 
# These steps collectively enable the clustering analysis of the Iris dataset, providing insights into how the data is structured and how individual data points can be grouped into meaningful clusters based on their features.
