#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:blue">
# <header>
# <h1 style="padding:1em;text-align:center;color:#00008B">Python programming <br><br> &nbsp;&nbsp; Pandas Library </h1> 
# </header>
# <br><br><br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# <header>
# <h1 style="padding:1em;text-align:left;color:#00008B">Outline</h1>
# </header>
# 
#     
# <ul><li><span style="color:#00008B;font-size:24px">What is Pandas? </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Importing Data </span> <br><br>
#         <li><span style="color:#00008B; font-size:24px">Analysing Data </span><br><br>
#         <li><span style="color:#00008B; font-size:24px">Visualising Data </span> 
#         </li>
#        </ul>
# 
# <br><br><span style="color:#00008B;font-size:20px">
#  <b>Note</b> - Using Python Libraries will allow you to analyse big datasets. Pandas is a must in the data science world, it is the most popular and widely used python librarie for data science along with NumPy, which is a mathematical python library.</span><br>
# <span style="color:#00008B;font-size:16px">
# Other Libraries you can use for data analysis. <a href="https://www.simplilearn.com/top-python-libraries-for-data-science-article">Python Libraries</a>
#     </span>
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Pandas Library</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Pandas can be applied in many different ways. <br><br>
# Loading and Saving Data <br><br>   
# Data Cleansing. <br><br>
# Data Inspection. <br><br>
# Data Visualisation. <br><br>
# 
# <a href="https://pandas.pydata.org/docs/user_guide/index.html">Pandas Documentation and User Guide</a>
# 
# 
# <h3 style="padding:1em;text-align:left;color:#00008B">Importing Pandas</h3>
# 
# import pandas as pd <br><br>
# 
# </span></li></ul> <br><br>
#  
# 
# <br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# ### Pandas Library in Python
# 
# **What is Pandas?**
# Pandas is a powerful and easy-to-use library for data manipulation and analysis in Python. It provides data structures like DataFrames and Series, which are ideal for handling structured data.
# 
# ### Key Features
# - **DataFrames**: 2-dimensional labeled data structure, similar to a table in a database or an Excel spreadsheet.
# - **Series**: 1-dimensional labeled array, useful for storing data along a single column or row.
# - **Data Manipulation**: Functions for filtering, grouping, aggregating, and merging data.
# - **Data Cleaning**: Tools for handling missing data, duplicates, and transforming data.
# - **Input/Output**: Easily read and write data from/to CSV, Excel, SQL databases, and more.
# 
# ### When to Use Pandas?
# - **Data Analysis**: When you need to explore and analyze data, perform statistical operations, or generate summary statistics.
# - **Data Cleaning**: When you have raw data that requires cleaning and preprocessing before analysis.
# - **Data Transformation**: When you need to transform data, like merging datasets, reshaping data, or handling time series data.
# - **Data Visualization**: When you want to quickly plot data for exploratory data analysis (often used with libraries like Matplotlib or Seaborn).
# 
# ### When Not to Use Pandas?
# - **Large Datasets**: For extremely large datasets that do not fit into memory, consider using libraries like Dask or PySpark.
# - **Performance-Critical Applications**: When you need high-performance operations, consider using NumPy or other optimized libraries, as Pandas can be slower due to its high-level functionality.
# - **Simple Calculations**: For basic mathematical operations on small datasets, using pure Python or NumPy might be simpler and faster.
# 
# ### Top points
# - **Pandas**: Ideal for data manipulation, analysis, and cleaning.
# - **Use for**: Exploratory data analysis, data preprocessing, handling structured data.
# - **Avoid for**: Large-scale data processing, performance-critical tasks, simple operations.

# In[ ]:


# Try importing pandas -> code below


# In[1]:


import pandas as pd


# <div style="background-color:beige;color:beige">
# 
# <h2 style="padding:1em;text-align:left;color:#00008B">Importing Data</h2>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# With Pandas you can import multiple types of data. <br><br>
# Such as CSV files, Excel files, SQL databases, JSON files, HTML tables, Parquet files, HDF5 files, Feather format, MessagePack, Stata files, SAS files, SPSS files, Pickle files, ORC files, BigQuery.<br><br>
# First step is identifying your file type.<br><br>   
# CSV or XLS. <br><br>
# Copy file path (shift right click). <br><br>
# Remeber to name your dataset in Jupyter. I will name mine 'data'.<br><br>
# Load your dataset: data = pd.read_csv(r'').<br><br>
#     
# How to create a Dataframe with pandas: df = pd.DataFrame(data=[define,your,data], columns = ['define', 'your','columns'], index=['define','your','indexes'])
# 
# Databases for datasets:
#       <li> <a href="https://www.kaggle.com/datasets">Kaggle </li> <br>
#       <li> <a href="https://archive.ics.uci.edu/ml/datasets/Iris">Machine Learning Repository</a><br>
# 
# </span></li></ul> <br><br>
#  
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[2]:


#Importing Iris dataset - > try here
iris_dataset = pd.read_csv(r'/Users/farah/Desktop/iris.csv')


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">ANALYSING DATA</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Understanding Basic information<br>
# <ul>
#   <li>.info() - Will return basic information about the dataframe </li>
#   <li>.shape - Will return the number of rows and columns in the dataframe </li>
#   <li>.head() - Will return the first 5 rows of the dataframe</li>
#   <li>.tail() - Will return the last 5 rows of the dataframe</li>
# </ul>
# <br><br>    
# Example: data.info()
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# ### Exercise Time

# In[3]:


# Try .info()
iris_dataset.info()


# In[4]:


# Try .shape
# Rows first, then column in the output
iris_dataset.shape


# In[5]:


# Try .head()

iris_dataset.head()


# In[6]:


# Try .tail()
iris_dataset.tail()


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Columns and indexes.</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Using .columns you will identify what columns your dataframes has.
#     
# Using .index you will identify what indexes your dataframe has.<br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[7]:


# Try below - What are the dataframes Columns?

iris_dataset.columns


# In[8]:


# Try below - what are the dataframes Indexes?

iris_dataset.index


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Descriptive Statistics</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Using .describe() will enable you to identify a quick statistic summary of your dataset.<br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[9]:


# Try .describe() 

iris_dataset.describe()


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">What if you have null values?</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Using .isnull().sum() will allow you to identify if your dataset has any null values or nan.<br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[10]:


# Try .isnull().sum()

iris_dataset.isnull().sum()


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Selecting a specific position?</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
#     
# Using .iloc you will be able select a specific position in your dataframe. You need to specify the index position. 
# 
# *Remember - The indexing rule (access other files to remind yourself).  
#     
# Example: data.iloc[1]  will select a single row in the dataset
# 
# Example: data.iloc[:, 0]  will select a single column in the dataset
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# ### Exercise Time - Try Isolating your data
# 
# 

# In[11]:


# Select rows between 1 and 19
iris_dataset.iloc[1:20]


# In[12]:


# Select rows between 15 and 25
iris_dataset.iloc[15:26]


# In[13]:


# Select only the last two colums

iris_dataset.iloc[:,3:5]


# In[14]:


# A bit more Challenging - Select only column 2 and 4 (Sepal Width and Petal Width)

iris_dataset.iloc[:,1:4:2] 
# skipping would come at the end.


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Other Important functions you can use with pandas.</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# .copy() - will copy your dataframe<br><br>
# .concat() - will concatenate two dataframes<br><br>
# .dropna() - will drop all rows with nan values<br><br>
# .mean() - will give you the mean of whatever you have selected<br><br>
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Exercise Time.</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# With the information you have gained so far. Please give me the mean of the column petal length. <br><br>
# 
# *Remember you can rename your new datasets<br><br>
# 
# Example: data1 = data.copy()
# 
# 
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>
# 

# In[ ]:


# With the information you have gained so far. Please give me the mean of the column petal length.


# In[15]:


iris_dataset_copy = iris_dataset.copy()
iris_dataset_copy


# In[18]:


# Step 1: Print the column names to check for the correct name
print(iris_dataset_copy.columns)


# In[17]:


# Step 2: Use the correct column name to calculate the mean
# The correct name is 'petal_length' after checking
petal_length_mean = iris_dataset_copy['petal_length'].mean()
print("Mean of petal length:", petal_length_mean)


# 
# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Exercise Time.</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Q1 Using .iloc() and .copy() <br><br>
# I want you to split the dataset in two. Use .copy() to not alter your original dataset. <br><br>
# *Remember to name your two new datasets<br><br>
# 
# Hint for Q2: data1 = data.concat([dataframe1,dataframe2], axis=1)
# 
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>
# 
# 

# In[19]:


# Q1 - Try Here - split the dataset in two, not loosing any value

number_one = iris_dataset_copy.iloc[1:4]
number_one


# In[20]:


number_two = iris_dataset_copy.iloc[50:120]
number_two


# In[21]:


# Q2 - Now concatenate the two dataframes into Iris_2 a new dataframe. hint use .concat()
# This comes with an error - reason? (below)
iris_2 = iris_dataset_copy.concat([number_one and number_2])
iris_2


# - The error is because `concat()` is not a method of individual DataFrame objects like `iris_dataset_copy`. 
# - Instead, `concat()` is a function provided by Pandas `pd.concat()` specifically designed to concatenate or join multiple DataFrames together along rows or columns. 
# - It's used to combine DataFrames into a single DataFrame, preserving the original structure of the individual DataFrames.
# - So, one should use `pd.concat()` because it's the correct way to concatenate DataFrames in Pandas, allowing you to efficiently combine data from multiple sources or parts of a dataset.

# In[22]:


iris_2 = pd.concat([number_one, number_two])
iris_2


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Renaming Columns</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# Using .rename() will allow you to rename either index or columns <br><br>
# You habe to specify which ones<br><br>
# 
# Example: data.rename(columns= {'Original column name':'New columns name'}, inplace=True)
# 
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>
# 

# In[23]:


print (iris_dataset_copy)


# In[27]:


# Try renaming columns to (SL, SW, PL, PW)
iris_dataset_copy.rename(columns={'sepal_length': 'SL', 'sepal_width': 'SW', 'petal_length': 'PL', 'petal_width': 'PW'}, inplace=True)

# Print the DataFrame to verify
print(iris_dataset_copy.head())


# ### Question:
# For each length type:
# Find out the average length for each species, Then figure out for each length type, which species has the highest average?

# In[32]:


# Group by species and calculate the mean
avg_lengths_species = iris_dataset_copy.groupby('species').mean()

# Display the average length for each species
print("Average length for each species:")
print(avg_lengths_species)


# In[33]:


# Group by species and calculate the mean
avg_lengths_species = iris_dataset_copy.groupby('species').mean()

# Find the species with the highest average length for each length type
highest_avg_species = avg_lengths_species.idxmax()

# Display the results
print("Species with the highest average length for each length type:")
print(highest_avg_species)


# ### Explanation:
# 1. **Group by Species**: `iris_dataset_copy.groupby('species').mean()` groups the DataFrame by species and calculates the mean of each numerical column (`SL`, `SW`, `PL`, `PW`).
# 
# 2. **Find Species with Highest Average Length**:
#    - `avg_lengths.idxmax()` finds the index (species name) with the highest value in each column.
#    - This will give you a Series where each index represents a type of length (e.g., 'SL', 'SW', 'PL', 'PW') and each value is the species with the highest average length for that type.
# 
# 3. **Print Results**:
#    - `print(highest_avg_lengths)` prints the species with the highest average length for each type of length.

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Graphical Analysis</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# 
# Visualisation Libraries<br>
# <ul>
#   <li> pandas </li>
#   <li> seaborn </li>
#   <li> matplotlib</li>
# </ul>
#     
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>
# 

# In[28]:


# to plot a simple graph use .plot() with your dataframe - Try below
iris_dataset_copy.plot()
iris_dataset_copy


# - The `.plot()` function in Pandas generates a simple line plot by default when applied directly to a DataFrame. 
# - This might not be the most meaningful representation for all types of data, especially when working with a DataFrame that contains both numeric and categorical data like the Iris dataset.
# 
# 
# To plot a meaningful graph, it's important to select appropriate columns and plot types. For example, plotting numerical columns like 'SL' (sepal length), 'SW' (sepal width), 'PL' (petal length), and 'PW' (petal width) against each other or against the index would make more sense. Here’s how you can plot these columns:
# 
# ### Explanation:
# 1. **Scatter Plot**: `.plot()` with `kind='scatter'` is used to create scatter plots, which are suitable for comparing two numeric variables.
# 
# 2. **Plotting Example**: The first plot (`Sepal Length vs Sepal Width`) plots 'SL' against 'SW', and the second plot (`Petal Length vs Petal Width`) plots 'PL' against 'PW'.
# 
# 3. **Matplotlib**: Matplotlib is used to customize the plot by adding titles, labels, and showing the plot.
# 
# ### Output:
# - Scatter plot of Sepal Length vs Sepal Width
# - Scatter plot of Petal Length vs Petal Width

# - `import matplotlib.pyplot as plt`: This line imports the `pyplot` module from the `matplotlib` library under the alias `plt`, which provides a MATLAB-like plotting framework.

# In[30]:


import matplotlib.pyplot as plt

# Plotting numerical columns against each other
fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

# Plot 1: Sepal Length vs Sepal Width
iris_dataset_copy.plot(x='SL', y='SW', kind='scatter', ax=axes[0], title='Sepal Length vs Sepal Width')
axes[0].set_xlabel('Sepal Length')
axes[0].set_ylabel('Sepal Width')

# Plot 2: Petal Length vs Petal Width
iris_dataset_copy.plot(x='PL', y='PW', kind='scatter', ax=axes[1], title='Petal Length vs Petal Width')
axes[1].set_xlabel('Petal Length')
axes[1].set_ylabel('Petal Width')

# Adjust layout to prevent overlapping - you can also skip this
plt.tight_layout()

# Save the figure as PNG format (you can change the format as needed)
plt.savefig('/Users/farah/Desktop/iris_plots.png')

# Display the plot
plt.show()


# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Graphical Analysis</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# 
# Graphical Analysis<br>
# <ul>
#   <li> Using Sublpots will automatically create plots according to the amount of columns you have.</li>
#   <li> Example: data.plot(subplots=True, figsize=(8, 8))</li>
# </ul>
#     
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# In[35]:


import matplotlib.pyplot as plt

# Create subplots for each column
iris_dataset_copy.plot(subplots=True, figsize=(10, 10))

# Adjust layout to prevent overlapping
plt.tight_layout()

# Specify the file path where you want to save the plots
file_path = '/Users/farah/Desktop/iris_subplots.png'

# Save the figure as PNG format
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Explanation:
# - **Create Subplots**: `iris_dataset_copy.plot(subplots=True, figsize=(10, 10))` creates subplots for each column in the DataFrame.
#   - `subplots=True` creates separate subplots for each column.
#   - `figsize=(10, 10)` sets the figure size to 10x10 inches.
# 
# - **Adjust Layout**: `plt.tight_layout()` adjusts the layout to prevent overlapping of subplots.
# 
# - **Specify File Path**: `file_path = '/Users/your_username/Desktop/iris_subplots.png'` specifies the file path where you want to save the plots. Replace `'your_username'` with your actual Mac username.
# 
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_subplots.png' at the specified file path.
# 
# - **Show the Plot**: `plt.show()` displays the plot on your screen.

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Other plots with pandas</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# 
# Graphical Analysis<br>
# <ul>
#   <li> Bar Graph - Example: df.plot(kind='bar')</li>
#   <li> Histogram - Example: df.plot.hist()</li>
#   <li> Boxplot - Example: df.plot.box() or df.boxplot()</li>
#   <li> Scatter plot - Example: df.plot.scatter(x='define', y='define')</li>
# </ul>
#     
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# __Now you try!__

# ### 1. Bar Graph
# A bar graph can be used to show the distribution of a categorical variable.

# In[36]:


#Try plotting a Bargraph

import matplotlib.pyplot as plt

# Bar graph
iris_dataset_copy['species'].value_counts().plot(kind='bar', figsize=(8, 6))
plt.title('Species Distribution')
plt.xlabel('Species')
plt.ylabel('Count')

# Save the figure
file_path = '/Users/farah/Desktop/iris_species_bar.png'
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Explanation:
# - **Bar Graph**: `iris_dataset_copy['species'].value_counts().plot(kind='bar', figsize=(8, 6))` creates a bar graph showing the count of each species.
#   - `kind='bar'` specifies the type of plot as a bar graph.
#   - `figsize=(8, 6)` sets the figure size to 8x6 inches.
# - **Title and Labels**: `plt.title('Species Distribution')`, `plt.xlabel('Species')`, `plt.ylabel('Count')` set the title and axis labels.
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_species_bar.png' to your desktop.
# - **Show the Plot**: `plt.show()` displays the plot.

# ### 2. Histogram
# A histogram is used to visualize the distribution of a numerical variable.

# In[37]:


#Try plotting a Histogram

# Histogram
iris_dataset_copy['PL'].plot.hist(figsize=(8, 6))
plt.title('Petal Length Distribution')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# Save the figure
file_path = '/Users/farah/Desktop/iris_petal_length_hist.png'
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Explanation:
# - **Histogram**: `iris_dataset_copy['PL'].plot.hist(figsize=(8, 6))` creates a histogram of the 'PL' (Petal Length) column.
# - **Title and Labels**: `plt.title('Petal Length Distribution')`, `plt.xlabel('Petal Length (cm)')`, `plt.ylabel('Frequency')` set the title and axis labels.
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_petal_length_hist.png' to your desktop.
# - **Show the Plot**: `plt.show()` displays the plot.

# ### 3. Boxplot
# A boxplot is used to show the distribution of numerical data in quartiles.

# In[38]:


#Try plotting a Boxblot

# Boxplot
iris_dataset_copy[['SL', 'SW', 'PL', 'PW']].plot.box(figsize=(10, 8))
plt.title('Boxplot of Sepal and Petal Dimensions')
plt.ylabel('Length (cm)')

# Save the figure
file_path = '/Users/farah/Desktop/iris_boxplot.png'
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Explanation:
# - **Boxplot**: `iris_dataset_copy[['SL', 'SW', 'PL', 'PW']].plot.box(figsize=(10, 8))` creates a boxplot of the numerical columns 'SL', 'SW', 'PL', 'PW'.
# - **Title and Labels**: `plt.title('Boxplot of Sepal and Petal Dimensions')`, `plt.ylabel('Length (cm)')` set the title and axis labels.
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_boxplot.png' to your desktop.
# - **Show the Plot**: `plt.show()` displays the plot.

# ### 4. Scatter Plot
# A scatter plot is used to show the relationship between two numerical variables.

# In[39]:


#Try plotting a Scatterplot with column 1 and 3

# Scatter plot
iris_dataset_copy.plot.scatter(x='PL', y='PW', figsize=(8, 6))
plt.title('Scatter Plot of Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# Save the figure
file_path = '/Users/farah/Desktop/iris_scatter_plot.png'
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Explanation:
# - **Scatter Plot**: `iris_dataset_copy.plot.scatter(x='PL', y='PW', figsize=(8, 6))` creates a scatter plot of 'PL' (Petal Length) vs 'PW' (Petal Width).
# - **Title and Labels**: `plt.title('Scatter Plot of Petal Length vs Petal Width')`, `plt.xlabel('Petal Length (cm)')`, `plt.ylabel('Petal Width (cm)')` set the title and axis labels.
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_scatter_plot.png' to your desktop.
# - **Show the Plot**: `plt.show()` displays the plot.

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Exercise</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# 
# Graphical Analysis exercise of each columns<br>
# 
# I want you to use all the information you have learnt so far. 
# 
# Task: Plot a bar graph of the means of each column in one graph. Note: not including the last one (species)
# <ul>
#   <li> Use the hints at your disposal</li>
#   <li> Don't be afraid to refer to previous slides</li>
#   <li> That's where the answer lies</li>
#   <li> GOOD LUCK! this is a more challenging exercise, but this will definelty help you with manipulating your own datasets - REMEMBER TO TAILOR IT TO YOUR DATASET!!!</li>
# </ul>
#      
# <br><br>
# 
# </span></li></ul>
#     
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# __Task: Plot a bar graph of the means of each column in one graph.__

# In[ ]:


# Hint 1: Isolate the 4 columns and re-define them.


# In[ ]:


#Hint 2: Use .mean() function to each of the columns.


# In[ ]:


#Hint 3: Refer to slide 4 to see how to create a dataframe. (Look carefully how the quote marks are placed)


# In[ ]:


# Plot your bar graph below using the code-guide: 


# In[40]:


# Import to make sure everything works smoothly

import pandas as pd
import matplotlib.pyplot as plt

# Isolate the 4 columns and compute means
means = iris_dataset_copy[['SL', 'SW', 'PL', 'PW']].mean()

# Create a DataFrame for means
means_df = pd.DataFrame({
    'Feature': means.index,
    'Mean Value': means.values
})

# Plot a bar graph
means_df.plot(x='Feature', y='Mean Value', kind='bar', figsize=(10, 6))
plt.title('Mean Values of Sepal and Petal Dimensions')
plt.xlabel('Feature')
plt.ylabel('Mean Value')

# Save the figure to your desktop
file_path = '/Users/farah/Desktop/iris_means_bar.png'
plt.savefig(file_path)

# Show the plot
plt.show()


# ### Steps to Achieve This:
# 
# 1. **Isolate the 4 Columns and Compute Means:**
#    - Extract the columns 'SL' (Sepal Length), 'SW' (Sepal Width), 'PL' (Petal Length), and 'PW' (Petal Width).
#    - Compute the mean of each column using the `.mean()` function.
# 
# 2. **Create a DataFrame for Means:**
#    - Create a new DataFrame with the means of each column.
#    - Set the column names as 'SL', 'SW', 'PL', and 'PW'.
# 
# 3. **Plot a Bar Graph:**
#    - Use `.plot(kind='bar')` to create a bar graph of the means of each column.
# 
# 4. **Save the Plot:**
#    - Save the plot to your desktop.
# 
# ### Explanation:
# - **Isolate the 4 Columns and Compute Means**: `iris_dataset_copy[['SL', 'SW', 'PL', 'PW']].mean()` selects the columns 'SL', 'SW', 'PL', and 'PW' from `iris_dataset_copy` and computes their means.
# 
# - **Create a DataFrame for Means**: `pd.DataFrame({...})` creates a new DataFrame `means_df` with columns 'Feature' (column names) and 'Mean Value' (mean values).
# 
# - **Plot a Bar Graph**: `means_df.plot(x='Feature', y='Mean Value', kind='bar', figsize=(10, 6))` creates a bar graph of mean values where x-axis is 'Feature' and y-axis is 'Mean Value'.
# 
# - **Save the Figure**: `plt.savefig(file_path)` saves the figure as 'iris_means_bar.png' to your desktop. Replace `'your_username'` with your actual Mac username.
# 
# - **Show the Plot**: `plt.show()` displays the plot.

# In[41]:


# iris_dataset_copy is your DataFrame
iris_dataset_copy.to_csv('/Users/farah/Desktop/iris_dataset_copy.csv', index=False)

