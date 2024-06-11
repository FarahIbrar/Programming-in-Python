#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:beige;color:blue">
# <header>
# <h1 style="padding:1em;text-align:center;color:#00008B">Basics of Python programming <br><br> &nbsp;&nbsp; Pandas Library </h1> 
# </header>
# <br><br><br><br><br><br>
# <footer>
# <table style="border:none;width:100%">
# <tr><td style= "text-align:left;background-color:blue;color:white;font-size:80%;"> Farah Ibrar </td><td style= "text-align:right;background-color:blue;color:white;font-size:80%;"> Programming in Python </td></tr></table>
# </footer>
# 
# </div>

# <div style="background-color:beige;color:beige">
# 
# <h1 style="padding:1em;text-align:center;color:#00008B">Code with me</h1>
# 
# <ul><li><span style="color:#00008B;font-size:20px">
# 
# 
# Other Graphical Libraries<br>
# 
# <ul>
#     <li> <a href='https://seaborn.pydata.org/'>Seaborn</a></li>
#     <li> <a href='https://matplotlib.org/'>Matplotlib</a></li>
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

# In[1]:


# Import seaborn as sns and matplotlib.pyplot as plt

import seaborn as sns 
import matplotlib.pyplot as plt


# ## Visualising tartget columns

# In[11]:


import pandas as pd
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')


# In[13]:


sns.countplot(x='species', data=iris_dataset, )
plt.show()


# In[14]:


# Save the figure
file_path = '/Users/farah/Desktop/iris_species_count.png'
plt.savefig(file_path)


# ## Comparing Sepal Length and Sepal Width

# ### This is a big one, let's start over.

# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Print the column names to verify, you can remove this as well if you remember and it won't show in the output
print(iris_dataset.columns)

# Now plot the scatter plot
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_dataset)

# Place legend outside the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Set title and labels
plt.title('Scatter plot of sepal_length vs. sepal_width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Save the figure
file_path = '/Users/farah/Desktop/sepal_scatterplot.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# ### Explanation:
# 1. **Import Libraries**: Import necessary libraries `pandas`, `seaborn`, and `matplotlib.pyplot`.
#   
# 2. **Load the Dataset**: Use `pd.read_csv()` to load the dataset `iris.csv` located on your desktop.
# 
# 3. **Verify Column Names**: Print `iris_dataset.columns` to verify the column names. Ensure they match the names used in the scatter plot.
# 
# 4. **Rename Columns (if necessary)**: Use `iris_dataset.rename()` to rename the columns to match the names used in the scatter plot (`sepal_length` and `sepal_width`).
# 
# 5. **Plot the Scatter Plot**:
#    - Use `sns.scatterplot()` to create a scatter plot.
#    - Specify `x='sepal_length'`, `y='sepal_width'`, and `hue='species'`.
# 
# 6. **Legend Placement**: 
#    - Use `plt.legend()` to place the legend outside the figure. 
#    - `bbox_to_anchor=(1.05, 1)` places the legend outside the figure to the right, and `loc='upper left'` specifies the position of the legend.
# 
# 7. **Title and Labels**: 
#    - Set the plot title, x-axis label, and y-axis label using `plt.title()`, `plt.xlabel()`, and `plt.ylabel()`.
# 
# 8. **Save the Figure**: 
#    - `plt.savefig()` saves the plot as a PNG file to the specified file path (`file_path`). 
#    - `bbox_inches='tight'` ensures that the legend is included in the saved image.
# 
# 9. **Display the Plot**: 
#    - `plt.show()` displays the plot.
# 
# ### Saving the Plot:
# - Adjust `'your_username'` in `file_path` to your actual Mac/window username.
# - Change `'sepal_scatterplot.png'` to your preferred filename and format if needed.

# ## Try doing the same with comparing Petal length and Petal Width

# In[ ]:


#Hint call data.columns to see how the columns are written


# In[22]:


# Now plot the scatter plot for Petal Length vs. Petal Width
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris_dataset)

# Place legend outside the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Set title and labels
plt.title('Scatter plot of Petal Length vs. Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

# Save the figure
file_path = '/Users/farah/Desktop/petal_scatterplot.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# ### Explanation:
# - **Plotting**: Use `sns.scatterplot()` to create the scatter plot comparing Petal Length (`x='petal_length'`) and Petal Width (`y='petal_width'`) with different species distinguished by color (`hue='species'`).
#   
# - **Legend Placement**: Use `plt.legend()` to place the legend outside the figure (`bbox_to_anchor=(1.05, 1), loc='upper left'`).
# 
# - **Title and Labels**: Set the plot title, x-axis label, and y-axis label using `plt.title()`, `plt.xlabel()`, and `plt.ylabel()`.
# 
# - **Save the Figure**: Save the plot as a PNG file using `plt.savefig()`. Adjust the file path (`file_path`) to save to your desktop.
# 
# - **Display the Plot**: Finally, use `plt.show()` to display the plot.

# ## Some more Powerful visualisation techniques

# In[ ]:


#Remove hastag below -  to see a multivariate analysis

#sns.pairplot(data, hue='Species', height=2)


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# The dataset is already loaded into iris_dataset frame

# Plot pairplot
sns.pairplot(iris_dataset, hue='species', height=2)

# Save the figure
file_path = '/Users/farah/Desktop/pairplot.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# ### `sns.pairplot()`
# 
# **What is `sns.pairplot()`?**
# - `sns.pairplot()` is a function from the Seaborn library used to create a grid of pairwise plots (scatter plots by default) for each pair of features in a dataset. It also supports various other kinds of plots on the diagonal and off-diagonal cells.
# 
# **Why do we use `sns.pairplot()`?**
# - **Visualization of relationships:** It helps us visualize relationships between variables in a dataset.
# - **Pattern identification:** It allows us to quickly identify patterns and correlations between variables.
# - **Diagnosing issues:** It's useful for diagnosing potential issues like outliers or skewness in the data.
# - **Facilitates exploration:** It facilitates data exploration and understanding of the dataset's structure.
# - **Conditional visualization:** With the `hue` parameter, we can visualize how different categories (e.g., species in this case) affect the relationships between variables.
# 
# **Parameters used:**
# - **`data`**: This is the dataset (DataFrame) that contains the variables to be plotted.
# - **`hue`**: Optional parameter. When specified, it colors the data points according to the categorical variable (e.g., species), making it easier to distinguish between different categories.
# - **`height`**: Optional parameter. It determines the height (in inches) of each facet. Default is 2 inches.
# 
# **Usage in the Context:**
# - In the provided code, `sns.pairplot(iris_dataset, hue='species', height=2)` is used to create a pairplot of the iris dataset.
# - Each pair of features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) is plotted against each other.
# - The `hue='species'` parameter is used to color the data points by the species of the iris flowers.
# - This visualization helps in understanding the relationship between different features and how they correlate across different species.
# 
# **Saving the Plot:**
# - `plt.savefig(file_path, bbox_inches='tight')`: Saves the plot as an image file named `pairplot.png` on your desktop. Adjust `file_path` to the desired location.
# 
# **Displaying the Plot:**
# - `plt.show()`: Displays the plot on the screen.
# 
# **Conclusion:**
# - `sns.pairplot()` is a powerful tool for initial exploratory data analysis (EDA) to understand the relationships and distributions in a dataset.
# - It helps in identifying patterns, correlations, and potential issues in the data.
# - The saved plot can be used for further analysis, reports, or presentations.
# 
# This function is particularly useful in the initial stages of data analysis to gain insights into the dataset before diving deeper into specific relationships or patterns.

# ## Coding a Displot

# __Distplot is used basically for the univariant set of observations and visualizes it through a histogram__

# In[ ]:


# add plt.show() at the end of the code to execute
plot = sns.FacetGrid(data, hue="species")
plot.map(sns.distplot, "sepal_length").add_legend()
 
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.distplot, "sepal_width").add_legend()
 
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.distplot, "petal_length").add_legend()
 
plot = sns.FacetGrid(data, hue="Species")
plot.map(sns.distplot, "petal_width").add_legend()


# It kept coming up with a lot of warnings so had to accomodate the code accordingly to what the warnings suggested:

# ## Sepal Length Distribution

# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Print the column names to verify
print(iris_dataset.columns)

# Plotting Sepal Length Distribution
plot = sns.FacetGrid(iris_dataset, hue="species")
plot.map(sns.histplot, "sepal_length").add_legend()

# Set title and labels
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')

# Save the figure
file_path = '/Users/farah/Desktop/sepal_length_distribution.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# All other coming analysis will follow the same code guide step-by-step, and I'll explain each part:
# 
# ```python
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# 
# # Load the dataset
# iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')
# 
# # Print the column names to verify
# print(iris_dataset.columns)
# ```
# 
# ### Step 1: Importing Libraries and Loading Data
# - **Pandas**: `import pandas as pd` is used to work with data in DataFrame format.
# - **Seaborn**: `import seaborn as sns` is used for statistical data visualization.
# - **Matplotlib**: `import matplotlib.pyplot as plt` is used for creating visualizations.
# 
# - **Loading Data**: `iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')` reads the CSV file into a pandas DataFrame named `iris_dataset`.
# 
# - **Printing Column Names**: `print(iris_dataset.columns)` prints the column names of the dataset to verify it was loaded correctly.
# 
# ### Step 2: Plotting Sepal Length Distribution
# ```python
# # Plotting Sepal Length Distribution
# plot = sns.FacetGrid(iris_dataset, hue="species")
# plot.map(sns.histplot, "sepal_length").add_legend()
# ```
# 
# - **FacetGrid**: `sns.FacetGrid(iris_dataset, hue="species")` initializes a Seaborn FacetGrid. The `hue="species"` parameter specifies that different species will be distinguished by color.
# 
# - **Mapping histplot**: `plot.map(sns.histplot, "sepal_length")` maps a histogram of `sepal_length` onto the grid.
# 
# - **Adding Legend**: `.add_legend()` adds a legend to the plot to distinguish between species.
# 
# ### Step 3: Setting Title and Labels
# ```python
# # Set title and labels
# plt.title('Distribution of Sepal Length')
# plt.xlabel('Sepal Length (cm)')
# plt.ylabel('Frequency')
# ```
# 
# - **Setting Title**: `plt.title('Distribution of Sepal Length')` sets the title of the plot.
# 
# - **Setting Labels**: `plt.xlabel('Sepal Length (cm)')` and `plt.ylabel('Frequency')` set the x-axis and y-axis labels, respectively.
# 
# ### Step 4: Saving the Figure
# ```python
# # Save the figure
# file_path = '/Users/farah/Desktop/sepal_length_distribution.png'
# plt.savefig(file_path, bbox_inches='tight')
# ```
# 
# - **Saving the Figure**: `plt.savefig(file_path, bbox_inches='tight')` saves the plot as an image file (`sepal_length_distribution.png`) to the specified file path (`/Users/farah/Desktop/`).
# 
# ### Step 5: Displaying the Plot
# ```python
# # Show the plot
# plt.show()
# ```
# 
# - **Displaying the Plot**: `plt.show()` displays the plot on the screen.
# 
# ### Explanation
# - **Purpose**: This code snippet is used to visualize the distribution of sepal lengths among different species of iris flowers.
# - **Why Use FacetGrid**: `sns.FacetGrid` is used to create multiple plots in a grid based on one or more variables (`hue="species"` in this case), making it easier to visualize relationships between variables.
# - **Why Use histplot**: `sns.histplot` is used to plot histograms, which are suitable for visualizing the distribution of a single continuous variable.
# - **Saving the Plot**: `plt.savefig()` is used to save the plot as a PNG file, which can be useful for including the visualization in reports or presentations.
# - **Legend**: The legend (`add_legend()`) is added to distinguish the different species, which is crucial when visualizing data that has a categorical variable like species.

# # Sepal Width Distribution

# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Sepal Width Distribution
plot = sns.FacetGrid(iris_dataset, hue="species")
plot.map(sns.histplot, "sepal_width").add_legend()

# Set title and labels
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')

# Save the figure
file_path = '/Users/farah/Desktop/sepal_width_distribution.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# #  Petal Width Distribution

# In[30]:


# Plotting Petal Width Distribution
plot = sns.FacetGrid(iris_dataset, hue="species")
plot.map(sns.histplot, "petal_width").add_legend()

# Set title and labels
plt.title('Distribution of Petal Width')
plt.xlabel('Petal Width (cm)')
plt.ylabel('Frequency')

# Save the figure
file_path = '/Users/farah/Desktop/petal_width_distribution.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# # Petal Length Distribution

# In[31]:


# Plotting Petal Length Distribution
plot = sns.FacetGrid(iris_dataset, hue="species")
plot.map(sns.histplot, "petal_length").add_legend()

# Set title and labels
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# Save the figure
file_path = '/Users/farah/Desktop/petal_length_distribution.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# ## Heatmap

# - to find the pairwise correlation of all columns in the dataframe.

# Let's proceed with creating a heatmap to visualize the pairwise correlation of all columns in the dataframe. 
# 
# We will use the `data.corr(method='pearson')` method to calculate the correlation matrix and then plot it using `sns.heatmap()`. 
# 
# Before we start, there's one thing to be aware of:
# 
# There might be a **warning** warning which basically means that it is indicating that the default value of `numeric_only` in the `DataFrame.corr` method is deprecated and will change in future versions of pandas. 
# 
# To **silence** this warning and ensure that only valid numeric columns are selected, we can explicitly set `numeric_only=True`.
# 
# If there is **no warning**, you can use the default settings and no need to add the addtional factor of `numeric_only=True`

# ### Calculate the Correlation Matrix
# 
# ```python
# import pandas as pd
# 
# # Load the dataset
# iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')
# 
# # Calculate the correlation matrix
# correlation_matrix = iris_dataset.corr(method='pearson', numeric_only=True)
# 
# # Print the correlation matrix to verify
# print(correlation_matrix)
# ```
# 
# ### Explanation:
# 1. **Loading the Dataset**: `pd.read_csv('/Users/farah/Desktop/iris.csv')` loads the iris dataset.
# 2. **Calculating the Correlation Matrix**: `iris_dataset.corr(method='pearson', numeric_only=True)` computes the Pearson correlation coefficients between the numeric columns, silencing the warning by explicitly setting `numeric_only=True`.
# 3. **Printing the Correlation Matrix**: `print(correlation_matrix)` prints the correlation matrix to the console for verification.

# In[ ]:


#data.corr(method='pearson')


# In[33]:


import pandas as pd

# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Calculate the correlation matrix
correlation_matrix = iris_dataset.corr(method='pearson', numeric_only=True)

# Print the correlation matrix to verify
print(correlation_matrix)


# ### Cell 2: Plotting the Heatmap
# 
# ```python
# # Plotting the heatmap
# plt.figure(figsize=(10, 8))  # Set the size of the heatmap
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# 
# # Set title and labels
# plt.title('Heatmap of Pairwise Correlation of Iris Dataset Columns')
# 
# # Save the figure
# file_path = '/Users/farah/Desktop/correlation_heatmap.png'
# plt.savefig(file_path, bbox_inches='tight')
# 
# # Show the plot
# plt.show()
# ```
# 
# ### Explanation:
# 1. **Setting the Figure Size**: `plt.figure(figsize=(10, 8))` sets the size of the heatmap plot.
# 2. **Plotting the Heatmap**: `sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")` creates the heatmap. 
#    - `correlation_matrix`: The data to plot.
#    - `annot=True`: Annotates each cell with the correlation coefficient.
#    - `cmap='coolwarm'`: Uses the 'coolwarm' colormap for better visualization of positive and negative correlations.
#    - `fmt=".2f"`: Formats the annotations to two decimal places.
# 3. **Setting Title**: `plt.title('Heatmap of Pairwise Correlation of Iris Dataset Columns')` sets the title of the heatmap.
# 4. **Saving the Figure**: `plt.savefig(file_path, bbox_inches='tight')` saves the plot as an image file (`correlation_heatmap.png`) to the specified file path (`/Users/farah/Desktop/`).
# 5. **Showing the Plot**: `plt.show()` displays the heatmap on the screen.

# In[ ]:


# sns.heatmap(data.corr(method='pearson'),annot = True);
 
# plt.show()


# In[34]:


#Make sure you have imported seaborn as sns and matplotlib.pyplot as plt

# Plotting the heatmap
plt.figure(figsize=(10, 8))  # Set the size of the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Set title and labels
plt.title('Heatmap of Pairwise Correlation of Iris Dataset Columns')

# Save the figure
file_path = '/Users/farah/Desktop/correlation_heatmap.png'
plt.savefig(file_path, bbox_inches='tight')

# Show the plot
plt.show()


# The heatmap is a data visualization technique that is used to analyze the dataset as colors in two dimensions.
# Shows a correlation between all numerical variables in the dataset.
