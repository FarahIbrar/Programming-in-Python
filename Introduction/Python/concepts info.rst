Concepts Information
====================

.. list-table::
   :header-rows: 1

   * - Concept
     - Description
     - When to Use
     - Typical Output
   * - Summary Statistics
     - Descriptive statistics that summarize the central tendency, dispersion, and shape of a dataset's distribution.
     - To get an overview of the dataset's numerical features.
     - Count, mean, standard deviation, min, 25th, 50th (median), 75th percentile, and max values for each numerical column.
   * - Pairwise Relationships (Pairplot)
     - A matrix of scatterplots and histograms to visualize relationships between pairs of variables.
     - To explore relationships between multiple variables simultaneously.
     - Scatter plots with histograms on the diagonal and scatter plots between variable pairs off the diagonal, often colored by a categorical variable.
   * - Box Plot
     - A method for graphically depicting groups of numerical data through their quartiles.
     - To compare distributions between different categories or groups.
     - A box with lines extending from it (whiskers) showing the variability outside the upper and lower quartiles, and individual points for outliers.
   * - Violin Plot
     - A combination of a box plot and a kernel density plot that shows the distribution of quantitative data across several levels of one (or more) categorical variables.
     - To compare distributions between different categories or groups, especially useful for comparing distribution shapes.
     - A kernel density estimation plot mirrored and joined to reflect the distribution shape, with a box plot inside showing quartiles and outliers.
   * - Swarm Plot
     - A categorical scatter plot with non-overlapping points.
     - To show all observations in a dataset.
     - Individual points spread out along the categorical axis, avoiding overlap.
   * - Joint Plot
     - A joint distribution plot combines information from scatter plots and histograms to provide detailed information for bivariate data.
     - To understand the relationship between two variables and the distribution of each separately.
     - A scatter plot with marginal histograms for two variables, optionally colored by a third variable.
   * - Pairwise KDE Plot
     - A kernel density estimate for bivariate data.
     - To visualize the density distribution between two variables.
     - A smooth heatmap-like plot showing the density of points between two variables, optionally colored by a third variable.
   * - FacetGrid
     - Multi-plot grid for plotting conditional relationships.
     - To visualize the distribution or relationship between variables within subsets of the dataset.
     - Multiple subplots showing relationships conditioned on different levels of a categorical variable.
   * - Boxen Plot
     - A box plot with more quantiles.
     - To visualize distributions especially in large datasets.
     - Similar to a box plot but with additional quantiles.
   * - ECDF Plot
     - Empirical cumulative distribution function plot.
     - To visualize the cumulative distribution of a single numerical variable.
     - A step function that jumps up by 1/N at each observed value of the variable.
   * - Rug Plot
     - A plot with small lines along an axis to denote the presence of individual data points.
     - To visualize the distribution of data points.
     - Small vertical lines or "ticks" along the x-axis or y-axis indicating the positions of data points.
   * - Scatter Plot Matrix
     - A grid of scatter plots for examining pairwise relationships in a dataset.
     - To visualize the relationship between every pair of numerical variables.
     - A matrix of scatter plots, with histograms on the diagonal, showing the relationship between each pair of variables.
   * - Andrews Curves
     - A method for visualizing data by mapping each observation to a function defined by a Fourier series.
     - To visualize multivariate data.
     - A plot where each observation is represented as a curve, with different classes shown in different colors.
   * - Parallel Coordinates
     - A method for plotting multivariate data.
     - To visualize and compare several numerical variables simultaneously.
     - A plot with multiple parallel lines, one for each variable, with lines connecting points across different categories or groups.
   * - RadViz
     - A method for visualizing multi-dimensional data in 2D.
     - To visualize multivariate data.
     - A plot where each variable is represented as a point on a circle, and each observation is represented as a point inside the circle.
   * - PCA Biplot
     - A scatterplot of data points using the first two principal components as axes.
     - To visualize the reduced dimensional representation of data after PCA.
     - A scatter plot where each point represents an observation, and the position of the point is determined by the first two principal components.
   * Missing Value Analysis
    - Description: Identifying and handling missing data in datasets.
    - When to Use: Before training a model to ensure all data is available.
    - Typical Output: Amount and distribution of missing values.
   * Unique Value Counts
    - Description: Counting distinct values in categorical variables.
    - When to Use: Understanding the diversity of categorical data.
    - Typical Output: Number of unique categories and their frequencies.
   * Species Distribution
    - Description: Distribution of classes or categories in a dataset.
    - When to Use: Assessing class balance before model training.
    - Typical Output: Proportions of different classes in the dataset.
   * Skewness and Kurtosis
    - Description: Measures of asymmetry and tail heaviness of a distribution.
    - When to Use: Assessing the shape of data distributions.
    - Typical Output: Degree and nature of deviation from normal distribution.
   * Normality Test (Shapiro-Wilk test)
    - Description: Statistical test to check if a variable is normally distributed.
    - When to Use: Before using parametric statistical methods.
    - Typical Output: Whether a variable follows a normal distribution.
   * Feature Scaling
    - Description: Scaling numerical features to a standard range.
    - When to Use: For algorithms sensitive to feature scaling.
    - Typical Output: Uniform range of values across different features.
   * Polynomial Features
    - Description: Generating polynomial combinations of features.
    - When to Use: Capturing non-linear relationships in data.
    - Typical Output: Enhanced model flexibility with polynomial terms.
   * Principal Component Analysis (PCA)
    - Description: Dimensionality reduction technique.
    - When to Use: Reducing the number of features while retaining variance.
    - Typical Output: New dimensions that explain the maximum variance.
  * Logistic Regression
    - Description: Linear model for binary classification.
    - When to Use: Predicting binary outcomes based on features.
    - Typical Output: Probability of class membership for each observation.
  * K-Nearest Neighbors (KNN)
    - Description: Instance-based learning for classification and regression.
    - When to Use: Classifying data based on similarity to known examples.
    - Typical Output: Class membership based on nearest neighbors.
  * Decision Tree Classifier
    - Description: Non-parametric supervised learning method.
    - When to Use: Predicting outcomes by learning simple decision rules.
    - Typical Output: Flowchart-like structure of decisions made.
  * Random Forest Classifier
    - Description: Ensemble learning method using multiple decision trees.
    - When to Use: Predicting outcomes with improved accuracy.
    - Typical Output: Combination of decision trees' predictions.
  * Support Vector Machine (SVM)
    - Description: Supervised learning model for classification and regression.
    - When to Use: Classifying data by finding an optimal hyperplane.
    - Typical Output: Decision boundary with maximum margin.
  * Cross-Validation
    - Description: Technique to evaluate predictive models.
    - When to Use: Assessing model performance with limited data.
    - Typical Output: Model performance metrics across different data subsets.
  * Hyperparameter Tuning (Grid Search)
    - Description: Method for optimizing model parameters.
    - When to Use: Maximizing model performance by tuning parameters.
    - Typical Output: Best set of parameters for optimal model performance.
  * Confusion Matrix
    - Description: Table showing true/false positive/negative predictions.
    - When to Use: Evaluating performance of classification models.
    - Typical Output: Breakdown of model's predictions versus actual outcomes.
  * Classification Report
    - Description: Summary of classification model's performance metrics.
    - When to Use: Assessing precision, recall, F1-score, and support.
    - Typical Output: Model's precision, recall, F1-score for each class.
  * Feature Importance
    - Description: Technique to identify most important features in a model.
    - When to Use: Understanding which features contribute most to predictions.
    - Typical Output: Ranking of features based on their importance.
  * ROC Curve
    - Description: Receiver Operating Characteristic curve.
    - When to Use: Evaluating binary classification model's performance.
    - Typical Output: Trade-off between true positive rate and false positive rate.
  * Clustering (K-Means)
    - Description: Unsupervised learning method to group data points.
    - When to Use: Discovering natural groupings in data.
    - Typical Output: Clusters of data points with similar characteristics.

