#!/usr/bin/env python
# coding: utf-8

# # Statistical Analysis

# In[2]:


import pandas as pd


# In[3]:


# Load the dataset
iris_dataset = pd.read_csv('/Users/farah/Desktop/iris.csv')

# Display the first few rows to verify
print(iris_dataset.head())


# In[4]:


# Variance of each feature
variance = iris_dataset.var()
print(variance)


# In[5]:


# Variance of each feature (using numeric only to avoid the potential warning - results would be same)
variance = iris_dataset.var(numeric_only=True)
print(variance)


# ## Variance
# 
# - **Explanation:** Measures the dispersion of the dataset.
# - **Why:** To understand how spread out the data points are.
# - **Shows:** The degree of variation in each feature.

# In[7]:


# Standard deviation of each feature
std_dev = iris_dataset.std(numeric_only=True)
print(std_dev)


# ## Standard Deviation 
# - **Explanation:** Measures the amount of variation or dispersion of a set of values. 
# - **Why:** To gauge the average distance of data points from the mean.
# - **Shows:** The extent of deviation for each feature.

# In[8]:


# Value counts for categorical features
species_counts = iris_dataset['species'].value_counts()
print(species_counts)


# ## Value Counts
# - **Explanation:** Counts the occurrences of each category in the 'species' column.
# - **Why:** To see the distribution of categorical data.
# - **Shows:** The frequency of each species in the dataset.

# In[ ]:


# Mode of each feature
mode_values = iris_dataset.mode()
print(mode_values)


# ## Mode
# - **Explanation:** Identifies the most frequent value in each column.
# - **Why:** To determine the most common data point.
# - **Shows:** The value that appears most often in each feature.

# In[ ]:


# IQR for each feature
Q1 = iris_dataset.quantile(0.25) # 25th percentile
Q3 = iris_dataset.quantile(0.75) # 75th percentile
IQR = Q3 - Q1
print(IQR)


# ## Interquartile Range (IQR)
# - **Explanation:** Measures the spread of the middle 50% of values.
# - **Why:** To understand the spread and detect outliers.
# - **Shows:** The range within which the central half of the data lies.

# - **Q1 (25th percentile):** This represents the value below which 25% of the data fall.
# - **Q3 (75th percentile):** This represents the value below which 75% of the data fall.
# - **IQR (Interquartile Range):** The range between Q1 and Q3, showing the spread of the middle 50% of the data.

# In[16]:


# Calculate the Interquartile Range (IQR) for each feature
Q1 = iris_dataset.quantile(0.25, numeric_only=True)  # 25th percentile
Q3 = iris_dataset.quantile(0.75, numeric_only=True)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range

# Create a DataFrame to align lower and upper bounds with the original DataFrame
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers using IQR
outliers = iris_dataset[
    (iris_dataset < lower_bound) | (iris_dataset > upper_bound)
].any(axis=1)
print(outliers)


# ## Outliers Detection using IQR
# - **Explanation:** Identifies outliers based on the IQR method.
# - **Why:** To find and potentially handle anomalous data points.
# - **Shows:** The data points that lie significantly outside the IQR.

# ### Explanation:
# - **Lower and Upper Bounds Calculation**: The lower and upper bounds are calculated using the IQR method to identify potential outliers.
# - **Comparison and Filtering**: The code checks for values in the `iris_dataset` that are below the lower bound or above the upper bound. If any such values are found in a row, that row is marked as having an outlier.
# - **Boolean Indexing**: The `.any(axis=1)` method is used to filter rows where any column contains an outlier. The result is a boolean Series indicating which rows have outliers.
# 
# ### Why We Need It and What It Shows:
# - **Outlier Detection**: Identifying outliers is crucial as they can distort statistical analyses and model predictions. This method helps in detecting anomalies that could indicate data entry errors, measurement errors, or unusual variations.
# - **IQR Method**: The IQR method is effective in identifying outliers by marking values that fall significantly outside the typical range, making it a robust tool for preliminary data analysis and ensuring data quality.

# In[18]:


# Coefficient of Variation
cv = iris_dataset.std(numeric_only=True) / iris_dataset.mean(numeric_only=True)
print(cv)


# ## Coefficient of Variation:
# - **Explanation:** Measures the relative variability.
# - **Why:** To compare the degree of variation from one dataset to another, regardless of units.
# - **Shows:** The ratio of the standard deviation to the mean for each feature.

# In[20]:


# Z-score standardization

import pandas as pd
from scipy.stats import zscore

# iris_dataset is already loaded as a DataFrame

# Select only numeric columns
numeric_columns = iris_dataset.select_dtypes(include=[float, int])

# Apply zscore function to the numeric columns
z_scores = numeric_columns.apply(zscore)

print(z_scores)


# ## Z-Score Standardization:
# - **Explanation:** Standardizes the dataset by converting values to z-scores.
# - **Why:** To normalize the data, making it easier to compare.
# - **Shows:** The number of standard deviations a data point is from the mean.

# ### Step-by-Step Breakdown
# 
# 1. **Import Required Libraries**:
#    ```python
#    import pandas as pd
#    from scipy.stats import zscore
#    ```
#    - `pandas` is used for data manipulation and analysis.
#    - `zscore` from `scipy.stats` is used for calculating Z-scores.
# 
# 2. **Assume `iris_dataset` is Loaded**:
#    ```python
#    # Assume iris_dataset is already loaded as a DataFrame
#    ```
#    - It is assumed that `iris_dataset` is already loaded into a pandas DataFrame.
# 
# 3. **Select Only Numeric Columns**:
#    ```python
#    numeric_columns = iris_dataset.select_dtypes(include=[float, int])
#    ```
#    - `select_dtypes(include=[float, int])` filters the DataFrame to include only columns with data types `float` and `int`. This ensures that only numeric columns are selected for Z-score standardization, avoiding errors with non-numeric columns.
# 
# 4. **Apply Z-score Function**:
#    ```python
#    z_scores = numeric_columns.apply(zscore)
#    ```
#    - `apply(zscore)` applies the `zscore` function to each numeric column in the filtered DataFrame.
#    - `zscore` standardizes each column by subtracting the mean and dividing by the standard deviation, resulting in columns with a mean of 0 and a standard deviation of 1.
# 
# 5. **Print the Resulting Z-scores**:
#    ```python
#    print(z_scores)
#    ```
#    - This prints the DataFrame of Z-scores for the numeric columns.

# In[21]:


# Binning continuous data
iris_dataset['sepal_length_bin'] = pd.cut(iris_dataset['sepal_length'], bins=3, labels=['Low', 'Medium', 'High'])
print(iris_dataset[['sepal_length', 'sepal_length_bin']].head())


# ## Binning
# - **Explanation:** Converts continuous data into categorical bins.
# - **Why:** To simplify analysis by categorizing continuous variables.
# - **Shows:** The distribution of data into specified bins.

# ### Step-by-Step Breakdown
# 
# 1. **Binning Continuous Data**:
#    ```python
#    iris_dataset['sepal_length_bin'] = pd.cut(iris_dataset['sepal_length'], bins=3, labels=['Low', 'Medium', 'High'])
#    ```
#    - `pd.cut` divides the `sepal_length` column into three equal-width bins.
#    - `bins=3` specifies that the data should be divided into three bins.
#    - `labels=['Low', 'Medium', 'High']` assigns labels to the bins. The lowest values are labeled 'Low', the middle values 'Medium', and the highest values 'High'.
# 
# 2. **Printing the First Five Rows**:
#    ```python
#    print(iris_dataset[['sepal_length', 'sepal_length_bin']].head())
#    ```
#    - This prints the first five rows of the DataFrame, showing the original `sepal_length` values and their corresponding bins.
# 
# ### Results
# 
# - The `sepal_length_bin` column categorizes the continuous `sepal_length` values into three categories: 'Low', 'Medium', and 'High'.
# - The printed output shows that the first five `sepal_length` values are all categorized as 'Low'. This indicates that these values fall into the lowest third of the `sepal_length` range. 

# In[23]:


# Covariance matrix
covariance_matrix = iris_dataset.cov(numeric_only=True)
print(covariance_matrix)


# ## Covariance matrix
# - **Explanation:** Measures how much two random variables vary together.
# - **Why:** To understand the relationship between different features.
# - **Shows:** The degree to which features change together.

# In[25]:


# Normalization of features

# Select only numeric columns
numeric_columns = iris_dataset.select_dtypes(include=[float, int])

# Perform normalization
normalized_data = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())

# Display the first few rows of the normalized data
print(normalized_data.head())


# ## Normalization
# - **Explanation:** Scales features to a range of [0, 1].
# - **Why:** To ensure features have the same scale for comparison.
# - **Shows:** The normalized values for each feature.

# ### Detailed Explanation
# 
# 1. **Select Only Numeric Columns**:
#    ```python
#    numeric_columns = iris_dataset.select_dtypes(include=[float, int])
#    ```
#    - This line filters the DataFrame to include only columns with numeric data types (`float` and `int`). This ensures that the subsequent arithmetic operations are only performed on numeric data.
# 
# 2. **Perform Normalization**:
#    ```python
#    normalized_data = (numeric_columns - numeric_columns.min()) / (numeric_columns.max() - numeric_columns.min())
#    ```
#    - `(numeric_columns - numeric_columns.min())` subtracts the minimum value of each column from every value in that column.
#    - `(numeric_columns.max() - numeric_columns.min())` calculates the range (max - min) for each column.
#    - Dividing these results normalizes the values to a [0, 1] range.
# 
# 3. **Display the First Few Rows of the Normalized Data**:
#    ```python
#    print(normalized_data.head())
#    ```
#    - This prints the first five rows of the normalized DataFrame, showing the normalized values for the numeric columns.
# 
# The output will be the first five rows of the normalized numeric columns, with values scaled to a range of 0 to 1. 
# This normalization ensures that all numeric features are scaled to the same range, which can be important for certain machine learning algorithms.

# In[26]:


# Chi-square test for categorical data
from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(iris_dataset['species'], iris_dataset['sepal_length_bin'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2: {chi2}, p-value: {p}")


# ## Chi-Square Test
# - **Explanation:** Tests the independence between categorical variables.
# - **Why:** To determine if there is a significant association between categorical features.
# - **Shows:** Whether the categorical variables are related.

# ### Detailed Explanation
# 
# 1. **Import Required Libraries**:
#    ```python
#    from scipy.stats import chi2_contingency
#    ```
#    - `chi2_contingency` is a function from the `scipy.stats` module that computes the chi-square statistic and p-value for the independence of the observed frequencies in a contingency table.
# 
# 2. **Create Contingency Table**:
#    ```python
#    contingency_table = pd.crosstab(iris_dataset['species'], iris_dataset['sepal_length_bin'])
#    ```
#    - `pd.crosstab` computes a cross-tabulation of two (or more) factors. In this case, it creates a contingency table where rows represent the species of iris and columns represent the bins of sepal lengths.
# 
# 3. **Perform Chi-square Test**:
#    ```python
#    chi2, p, dof, expected = chi2_contingency(contingency_table)
#    ```
#    - `chi2_contingency` is used to perform the chi-square test of independence on the contingency table.
#    - `chi2` is the test statistic, which follows a chi-square distribution.
#    - `p` is the p-value of the test, indicating the probability of observing the data under the null hypothesis of independence.
#    - `dof` is the degrees of freedom.
#    - `expected` is the expected frequencies under the null hypothesis.
# 
# 4. **Print Results**:
#    ```python
#    print(f"Chi2: {chi2}, p-value: {p}")
#    ```
#    - This line prints the computed chi-square statistic (`chi2`) and the p-value (`p`) obtained from the chi-square test.
# 
# ### Summary
# 
# - The chi-square test for independence evaluates whether there is a statistically significant association between the categorical variables (`species` and `sepal_length_bin` in this case).
# - The null hypothesis is that the two categorical variables are independent.
# - The chi-square statistic (`chi2`) and p-value (`p`) are computed and printed to assess whether to reject or fail to reject the null hypothesis.
# 
# This test is useful for determining if there is a relationship between two categorical variables, which is often important in the analysis of categorical data and in some types of statistical modeling.

# In[27]:


# T-test between two species for a feature
from scipy.stats import ttest_ind
setosa = iris_dataset[iris_dataset['species'] == 'setosa']['sepal_length']
versicolor = iris_dataset[iris_dataset['species'] == 'versicolor']['sepal_length']
t_stat, p_val = ttest_ind(setosa, versicolor)
print(f"T-statistic: {t_stat}, p-value: {p_val}")


# ## T-Test
# - **Explanation:** Compares the means of two independent groups.
# - **Why:** To test if the means of two groups are significantly different.
# - **Shows:** Whether there is a significant difference in 'sepal_length' between 'setosa' and 'versicolor'.

# In[28]:


# ANOVA test across species for a feature
from scipy.stats import f_oneway
setosa = iris_dataset[iris_dataset['species'] == 'setosa']['sepal_length']
versicolor = iris_dataset[iris_dataset['species'] == 'versicolor']['sepal_length']
virginica = iris_dataset[iris_dataset['species'] == 'virginica']['sepal_length']
f_stat, p_val = f_oneway(setosa, versicolor, virginica)
print(f"F-statistic: {f_stat}, p-value: {p_val}")


# ## ANOVA test
# - **Explanation:** Compares the means across multiple groups.
# - **Why:** To test if there are significant differences among group means.
# - **Shows:** Whether 'sepal_length' differs significantly across the species.

# In[33]:


# Kurtosis of each feature
kurtosis = iris_dataset.kurtosis
print(kurtosis) 


# In[34]:


# Kurtosis of each feature
kurtosis = iris_dataset.kurtosis (numeric_only=True)
print(kurtosis)


# ## Kurtosis
# - **Explanation:** Measures the "tailedness" of the distribution.
# - **Why:** To understand the extremity of outliers in the dataset.
# - **Shows:** How heavy or light-tailed the distributions are.

# In[35]:


# Covariance between two features
covariance = iris_dataset['sepal_length'].cov(iris_dataset['sepal_width'])
print(f"Covariance: {covariance}")


# ## Covariance Calculation
# - **Explanation:** Calculates the covariance between two specific features.
# - **Why:** To determine the relationship between two variables.
# - **Shows:** The extent to which 'sepal_length' and 'sepal_width' vary together.

# In[36]:


# Autocorrelation of a feature
autocorrelation = iris_dataset['sepal_length'].autocorr()
print(f"Autocorrelation: {autocorrelation}")


# ## Autocorrelation
# - **Explanation:** Measures the correlation of a feature with a lagged version of itself.
# - **Why:** To detect patterns or periodicity in time series data.
# - **Shows:** The degree of correlation of 'sepal_length' with itself at different time lags.

# In[37]:


# Cross-tabulation of two categorical features
crosstab_result = pd.crosstab(iris_dataset['species'], iris_dataset['sepal_length_bin'])
print(crosstab_result)


# ## Cross-Tabulation
# - **Explanation:** Creates a contingency table to analyze the relationship between two categorical features.
# - **Why:** To summarize the relationship between two categorical variables.
# - **Shows:** The frequency distribution of 'species' across 'sepal_length_bin'.

# In[39]:


# Log transformation of a feature

import numpy as np
import pandas as pd

iris_dataset['log_sepal_length'] = np.log(iris_dataset['sepal_length'])
print(iris_dataset[['sepal_length', 'log_sepal_length']].head())


# ## Log Transformation
# - **Explanation:** Applies a log transformation to reduce skewness.
# - **Why:** To handle skewed data and stabilize variance.
# - **Shows:** The transformed 'sepal_length' values with reduced skewness.
