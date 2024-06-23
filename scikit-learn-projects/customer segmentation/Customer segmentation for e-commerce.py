#!/usr/bin/env python
# coding: utf-8

# # Customer segmentation for e-commerce

# ## Step 1: Data Exploration and Preprocessing

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
df = pd.read_excel(url)


# In[3]:


# Display the first few rows of the dataset
df.head()


# In[4]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())


# In[5]:


# Drop rows with missing CustomerID, as we need this for segmentation
df.dropna(subset=['CustomerID'], inplace=True)


# In[6]:


# Drop duplicates if any
df.drop_duplicates(inplace=True)


# In[7]:


# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[8]:


# Feature engineering: Calculate total purchase amount
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']


# In[9]:


# Explore data distribution
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


# Distribution of Quantity
plt.figure(figsize=(10, 6))
sns.histplot(df['Quantity'], bins=30, kde=True)
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Count')
plt.grid(True)

# Save the plot as PNG in your Jupyter repository
plt.savefig('distribution_quantity.png')
plt.show()


# In[11]:


# Example: Monthly sales trends
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
monthly_sales = df.groupby('InvoiceMonth').size()

plt.figure(figsize=(12, 6))
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.grid(True)

# Save the plot as PNG in your Jupyter repository
plt.savefig('monthly_sales_trends.png')
plt.show()


# In[12]:


# Save cleaned data to a new file
df.to_csv('cleaned_data.csv', index=False)


# In[ ]:





# ## Step 2: Customer Segmentation using Clustering

# In[13]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[14]:


# Load cleaned data
df = pd.read_csv('cleaned_data.csv')


# In[15]:


# Select relevant features for clustering
X = df[['Quantity', 'UnitPrice', 'TotalAmount']]


# In[16]:


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[17]:


# Determine optimal number of clusters using the elbow method
distortions = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init to suppress warning (optional)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)


# In[18]:


# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')

# Save the elbow plot as PNG in your Jupyter repository
plt.savefig('elbow_plot.png')
plt.show()


# In[19]:


# Based on the elbow method, choose the optimal number of clusters (k)
k = 4  # Adjust this based on your analysis from the elbow plot


# In[20]:


# Perform K-means clustering with the chosen k
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_


# In[21]:


# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Quantity', y='TotalAmount', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8)
plt.title('Customer Segmentation by K-means Clustering')
plt.xlabel('Quantity')
plt.ylabel('Total Amount')
plt.legend(title='Cluster')

# Save the scatter plot as PNG in your Jupyter repository
plt.savefig('customer_segmentation.png')
plt.show()


# In[22]:


# Save the clustered data
df.to_csv('clustered_data.csv', index=False)


# In[ ]:





# ## Step 3: Customer Profiling and Insights

# In[23]:


# Load clustered data
df = pd.read_csv('clustered_data.csv')


# In[24]:


# General statistics for each cluster
cluster_stats = df.groupby('Cluster').agg({
    'CustomerID': 'nunique',
    'TotalAmount': ['mean', 'sum'],
    'InvoiceDate': ['min', 'max']
}).reset_index()


# In[25]:


cluster_stats.columns = ['Cluster', 'NumCustomers', 'AvgTotalAmount', 'TotalRevenue', 'MinPurchaseDate', 'MaxPurchaseDate']
print(cluster_stats)


# In[26]:


# Visualizing Average Total Amount per Cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='AvgTotalAmount', data=cluster_stats, palette='viridis')
plt.title('Average Total Amount per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Total Amount')

# Save the plot as PNG
plt.savefig('avg_total_amount_per_cluster.png')
plt.show()


# In[27]:


# Visualizing Number of Customers per Cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='NumCustomers', data=cluster_stats, palette='viridis')
plt.title('Number of Customers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')

# Save the plot as PNG
plt.savefig('num_customers_per_cluster.png')
plt.show()


# In[28]:


# Additional insights and visualizations
# For example, analyzing purchase frequency and recency


# In[29]:


# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[30]:


# Calculate Recency (days since last purchase)
df['Recency'] = (df['InvoiceDate'].max() - df['InvoiceDate']).dt.days

print(df)


# In[31]:


# Aggregate by CustomerID and Cluster
customer_agg = df.groupby(['CustomerID', 'Cluster']).agg({
    'TotalAmount': 'sum',
    'InvoiceNo': 'count',
    'Recency': 'min'
}).reset_index()


# In[32]:


customer_agg.columns = ['CustomerID', 'Cluster', 'TotalAmount', 'Frequency', 'Recency']


# In[33]:


# Average metrics per cluster
cluster_agg_stats = customer_agg.groupby('Cluster').agg({
    'TotalAmount': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean'
}).reset_index()


# In[34]:


cluster_agg_stats.columns = ['Cluster', 'AvgTotalAmount', 'AvgFrequency', 'AvgRecency']
print(cluster_agg_stats)


# In[35]:


# Visualizing Average Frequency per Cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='AvgFrequency', data=cluster_agg_stats, palette='viridis')
plt.title('Average Purchase Frequency per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Purchase Frequency')
plt.grid(False)

# Save the plot as PNG
plt.savefig('avg_purchase_frequency_per_cluster.png')
plt.show()


# In[36]:


# Visualizing Average Recency per Cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='Cluster', y='AvgRecency', data=cluster_agg_stats, palette='viridis')
plt.title('Average Recency (Days Since Last Purchase) per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Recency (Days)')
plt.grid(False)

# Save the plot as PNG
plt.savefig('avg_recency_per_cluster.png')
plt.show()

