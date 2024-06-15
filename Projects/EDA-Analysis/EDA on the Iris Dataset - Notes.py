#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) on the Iris Dataset

# #### Step 1: Data Summarization
# 
# ```python
# # Import necessary libraries
# import pandas as pd
# import seaborn as sns
# from sklearn.datasets import load_iris
# 
# # Load the Iris dataset
# iris = load_iris()
# iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_dataset['species'] = iris.target
# 
# # Mapping target values to actual species names
# iris_dataset['species'] = iris_dataset['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# 
# # Descriptive statistics
# desc_stats = iris_dataset.describe()
# print("Descriptive Statistics:\n", desc_stats)
# 
# # Correlation matrix
# corr_matrix = iris_dataset.corr(numeric_only=True)
# print("\nCorrelation Matrix:\n", corr_matrix)
# ```

# ### **Explanation**:
# - **Loading the Iris Dataset**:
#   - Load the Iris dataset using `load_iris()` from `sklearn.datasets`.
#   - Create a DataFrame with feature names and add a column for species.
#   - Map target values (0, 1, 2) to actual species names ('setosa', 'versicolor', 'virginica').
#   - **Why This Step is Needed**: This step prepares the dataset for analysis by adding meaningful labels to the data.
# 
# - **Descriptive Statistics**:
#   - Use `describe()` to get a summary of the central tendency, dispersion, and shape of the dataset’s distribution.
#   - **Why This Step is Needed**: Descriptive statistics provide a quick overview of the data, including mean, standard deviation, and range, which are essential for understanding the distribution of each feature.
# 
# - **Correlation Matrix**:
#   - Use `corr()` to calculate the correlation coefficients between features.
#   - **Why This Step is Needed**: Correlation analysis helps to identify relationships between features, which is useful for understanding feature interactions and potential multicollinearity issues.

# ### Step 2: Data Visualization
# 
# ```python
# # Import necessary libraries
# import matplotlib.pyplot as plt
# 
# # Histograms
# iris_dataset.hist(figsize=(10, 8))
# plt.suptitle('Histograms of Iris Dataset Features')
# 
# # Specify the file path where you want to save the figure
# file_path = '/Users/farah/Desktop/Python Workshop/Project/Project 2 - EDA on the Iris Dataset/EDA Images/iris_histograms.png'
# 
# # Save the figure
# plt.savefig(file_path, dpi=300)
# plt.show()
# 
# # Box plots
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=iris_dataset.drop(columns='species'))
# plt.title('Box Plots of Iris Dataset Features')
# plt.show()
# 
# # Scatter plots
# sns.pairplot(iris_dataset, hue='species')
# plt.suptitle('Pairplot of Iris Dataset Features')
# plt.show()
# ```

# ### **Explanation**:
# - **Histograms**:
#   - Plot histograms for each feature to visualize their distributions.
#   - **Why This Step is Needed**: Histograms provide a visual representation of the distribution of individual features, helping to identify skewness, modality, and potential outliers.
# 
# - **Box Plots**:
#   - Plot box plots for each feature to visualize their distributions, central tendency, and spread.
#   - **Why This Step is Needed**: Box plots help to identify the median, quartiles, and potential outliers for each feature, providing insights into their variability and distribution.
# 
# - **Scatter Plots (Pairplot)**:
#   - Use pair plots to visualize pairwise relationships between features, colored by species.
#   - **Why This Step is Needed**: Pair plots help to identify relationships and potential patterns between features, and how different species are distributed across feature pairs.

# ### Step 3: Feature Distribution Analysis Across Species
# 
# ```python
# # Violin plots to visualize feature distributions across species
# plt.figure(figsize=(15, 10))
# for i, feature in enumerate(iris_dataset.columns[:-1]):
#     plt.subplot(2, 2, i+1)
#     sns.violinplot(x='species', y=feature, data=iris_dataset)
#     plt.title(f'Violin Plot of {feature} by Species')
# plt.tight_layout()
# plt.show()
# ```

# ### **Explanation**:
# - **Violin Plots**:
#   - Plot violin plots for each feature to visualize their distributions across different species.
#   - **Why This Step is Needed**: Violin plots provide a detailed view of the distribution of features for each species, helping to understand the differences and similarities between species.

# ### Step 4: Hypothesis Testing
# 
# ```python
# # Import necessary library
# from scipy.stats import f_oneway
# 
# # ANOVA test to determine if there are significant differences between species
# features = iris_dataset.columns[:-1]
# anova_results = {feature: f_oneway(iris_dataset[iris_dataset['species'] == 'setosa'][feature],
#                                    iris_dataset[iris_dataset['species'] == 'versicolor'][feature],
#                                    iris_dataset[iris_dataset['species'] == 'virginica'][feature])
#                  for feature in features}
# 
# # Print ANOVA results
# for feature, result in anova_results.items():
#     print(f"ANOVA test for {feature}: F-statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4e}")
# ```

# ### **Explanation**:
# - **ANOVA Test**:
#   - Perform ANOVA (Analysis of Variance) test to determine if there are significant differences in feature means between species.
#   - **Why This Step is Needed**: ANOVA helps to identify whether the differences observed between species are statistically significant, providing a deeper understanding of the dataset.

# ### Step 5: Insights and Conclusions
# 
# ```python
# # Summarize insights and conclusions
# insights = """
# Insights and Conclusions:
# 1. Descriptive statistics reveal that petal length and petal width have the highest means and standard deviations.
# 2. Correlation analysis shows a strong positive correlation between petal length and petal width.
# 3. Histograms and box plots indicate that sepal width has a more uniform distribution, while petal features are more varied.
# 4. Scatter plots show clear separations between species, especially in petal features.
# 5. Violin plots reveal that 'setosa' species are distinct in all features, while 'versicolor' and 'virginica' overlap in some features.
# 6. ANOVA tests indicate significant differences between species for all features (p-value < 0.05).
# """
# 
# print(insights)
# ```

# ### **Explanation**:
# - **Insights and Conclusions**:
#   - Summarize the key findings from the EDA.
#   - **Why This Step is Needed**: Summarizing insights helps to consolidate the findings, making it easier to communicate the key points and conclusions derived from the analysis.
# 
# This completes the Exploratory Data Analysis (EDA) on the Iris dataset. Let me know if you want to proceed with another analysis or need further details on any of the steps!

# In[ ]:





# ### Interpretation of Results from EDA
# 
# 1. **Descriptive Statistics**:
#    - The descriptive statistics reveal that the mean and standard deviation of the petal length and petal width are significantly higher than those of the sepal length and sepal width. This suggests that the petal features vary more than the sepal features among the iris flowers.
# 
# 2. **Correlation Matrix**:
#    - The correlation matrix shows a strong positive correlation between petal length and petal width (0.96). This indicates that as the petal length increases, the petal width tends to increase as well. Other significant correlations include sepal length with petal length (0.87) and sepal length with petal width (0.82). These correlations suggest interdependencies between these features.
# 
# 3. **Histograms**:
#    - The histograms of the features show that petal length and petal width have more varied distributions compared to sepal length and sepal width. The distributions of petal features are more spread out, indicating greater variability.
# 
# 4. **Box Plots**:
#    - The box plots reveal that sepal width has a relatively uniform distribution, while petal features show more variability with noticeable outliers. This suggests that petal measurements are more diverse among the iris flowers.
# 
# 5. **Scatter Plots (Pairplot)**:
#    - The pair plots highlight clear separations between the species, especially in petal length and petal width. The 'setosa' species forms a distinct cluster separate from 'versicolor' and 'virginica', which overlap slightly. This indicates that petal features are useful for distinguishing between species.
# 
# 6. **Violin Plots**:
#    - The violin plots show that the 'setosa' species is distinct in all features, with its sepal and petal measurements forming clear, separate distributions from the other species. 'Versicolor' and 'virginica' overlap in some features, indicating that these two species are more similar to each other in terms of these measurements.
# 
# 7. **ANOVA Test**:
#    - The ANOVA test results indicate significant differences between species for all features, with p-values well below 0.05. This confirms that the differences in feature means between species are statistically significant, reinforcing the utility of these features for classification.
# 
# ### Summary of Insights and Conclusions
# 
# - The petal length and petal width are key distinguishing features among the iris species, showing higher variability and clearer separations.
# - The strong correlations between petal features suggest they are closely related and can be used together for species classification.
# - The distinct distributions of 'setosa' in both petal and sepal features highlight its uniqueness, while the overlap between 'versicolor' and 'virginica' suggests these species are more similar.
# - The ANOVA test confirms that all features are significantly different across species, validating their importance for differentiating between iris species.
