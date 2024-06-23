Concepts Information
====================

.. list-table::
   :header-rows: 1
   :widths: 20 50 20 30

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
   * - Missing Value Analysis
     - Identifying and handling missing data in datasets.
     - Before training a model to ensure all data is available.
     - Amount and distribution of missing values.
   * - Unique Value Counts
     - Counting distinct values in categorical variables.
     - Understanding the diversity of categorical data.
     - Number of unique categories and their frequencies.
   * - Species Distribution
     - Distribution of classes or categories in a dataset.
     - Assessing class balance before model training.
     - Proportions of different classes in the dataset.
   * - Skewness and Kurtosis
     - Measures of asymmetry and tail heaviness of a distribution.
     - Assessing the shape of data distributions.
     - Degree and nature of deviation from normal distribution.
   * - Normality Test (Shapiro-Wilk test)
     - Statistical test to check if a variable is normally distributed.
     - Before using parametric statistical methods.
     - Whether a variable follows a normal distribution.
   * - Feature Scaling
     - Scaling numerical features to a standard range.
     - For algorithms sensitive to feature scaling.
     - Uniform range of values across different features.
   * - Polynomial Features
     - Generating polynomial combinations of features.
     - Capturing non-linear relationships in data.
     - Enhanced model flexibility with polynomial terms.
   * - Principal Component Analysis (PCA)
     - Dimensionality reduction technique.
     - Reducing the number of features while retaining variance.
     - New dimensions that explain the maximum variance.
   * - Logistic Regression
     - Linear model for binary classification.
     - Predicting binary outcomes based on features.
     - Probability of class membership for each observation.
   * - K-Nearest Neighbors (KNN)
     - Instance-based learning for classification and regression.
     - Classifying data based on similarity to known examples.
     - Class membership based on nearest neighbors.
   * - Decision Tree Classifier
     - Non-parametric supervised learning method.
     - Predicting outcomes by learning simple decision rules.
     - Flowchart-like structure of decisions made.
   * - Random Forest Classifier
     - Ensemble learning method using multiple decision trees.
     - Predicting outcomes with improved accuracy.
     - Combination of decision trees' predictions.
   * - Support Vector Machine (SVM)
     - Supervised learning model for classification and regression.
     - Classifying data by finding an optimal hyperplane.
     - Decision boundary with maximum margin.
   * - Cross-Validation
     - Technique to evaluate predictive models.
     - Assessing model performance with limited data.
     - Model performance metrics across different data subsets.
   * - Hyperparameter Tuning (Grid Search)
     - Method for optimizing model parameters.
     - Maximizing model performance by tuning parameters.
     - Best set of parameters for optimal model performance.
   * - Confusion Matrix
     - Table showing true/false positive/negative predictions.
     - Evaluating performance of classification models.
     - Breakdown of model's predictions versus actual outcomes.
   * - Classification Report
     - Summary of classification model's performance metrics.
     - Assessing precision, recall, F1-score, and support.
     - Model's precision, recall, F1-score for each class.
   * - Feature Importance
     - Technique to identify most important features in a model.
     - Understanding which features contribute most to predictions.
     - Ranking of features based on their importance.
   * - ROC Curve
     - Receiver Operating Characteristic curve.
     - Evaluating binary classification model's performance.
     - Trade-off between true positive rate and false positive rate.
   * - Clustering (K-Means)
     - Unsupervised learning method to group data points.
     - Discovering natural groupings in data.
     - Clusters of data points with similar characteristics.
   * - Variance
     - Measures the dispersion of the dataset.
     - To understand how spread out the data points are in a dataset.
     - Numeric value indicating the squared deviation from the mean.
   * - Standard Deviation
     - Measures the amount of variation or dispersion of a set of values.
     - When you need to quantify the amount of variation or dispersion in a dataset.
     - Numeric value representing the average distance of data points from the mean.
   * - Numeric only
     - Data consisting only of numeric values.
     - When you need to operate on numerical data specifically.
     - Subset of the original dataset containing only numeric columns.
   * - Mode
     - Identifies the most frequent value in a dataset.
     - When you need to find the most common value or values in a dataset.
     - Single or multiple values representing the most frequent items in the dataset.
   * - Interquartile range (IQR)
     - Measures the spread of the middle 50% of values.
     - When you want to understand the spread of the central part of the data distribution.
     - Range of values that represents the spread of the central 50% of the data.
   * - Outliers
     - Data points that are significantly different from the majority of the data.
     - When you want to identify data points that may be erroneous or require special treatment.
     - List of values that fall far from the central tendency of the dataset.
   * - Coefficient of Variation
     - Measures the relative variability.
     - When comparing the variability of datasets with different means.
     - Numeric value representing the relative variability normalized to the mean.
   * - Z-Score Standardization
     - Standardizes data by converting values to z-scores.
     - When you want to compare data points that have different scales.
     - Standardized values (z-scores) that represent the deviation from the mean in terms of standard deviations.
   * - Bins
     - Categorizes continuous data into intervals or groups.
     - When you need to discretize continuous data.
     - Interval labels representing the categorized data points.
   * - Covariance matrix
     - Shows how much two random variables vary together.
     - When you need to analyze the relationship between multiple variables.
     - Matrix where entries represent covariances between pairs of variables.
   * - Normalization
     - Scales features to a range of [0, 1].
     - When you want to scale features to a uniform range.
     - Values of the dataset scaled to the interval [0, 1].
   * - Chi-square test
     - Tests the independence between categorical variables.
     - When you need to determine if two categorical variables are associated.
     - Chi-square statistic and p-value indicating the strength of association between variables.
   * - chi2_contingency
     - Computes the chi-square statistic and p-value for a cross-tabulation.
     - When you need to test the association between two categorical variables.
     - Chi-square statistic, p-value, degrees of freedom, and expected frequencies.
   * - T-Test
     - Compares the means of two independent groups.
     - When you need to determine if there is a significant difference between two groups.
     - T-statistic, p-value, and degrees of freedom.
   * - ttest_ind
     - Computes the T-test for the means of two independent samples.
     - When comparing two groups for significant differences.
     - T-statistic and p-value.
   * - ANOVA test
     - Compares the means of multiple groups.
     - When you need to determine if there are significant differences among multiple groups.
     - F-statistic, p-value, and degrees of freedom.
   * - f_oneway
     - Computes the one-way ANOVA.
     - When comparing means across multiple groups.
     - F-value and p-value.
   * - Kurtosis
     - Measures the "tailedness" of the distribution.
     - When you need to understand the shape of the distribution, particularly its tails.
     - Numeric value indicating the kurtosis of the dataset.
   * - Covariance
     - Measures how much two variables change together.
     - When you need to understand the directional relationship between two variables.
     - Numeric value indicating the covariance between two variables.
   * - Autocorrelation
     - Measures the correlation of a variable with a lagged version of itself.
     - When you need to understand if a variable is correlated with a previous value of itself.
     - Numeric values representing the autocorrelation at different lag intervals.
   * - Cross-tabulation
     - Creates a contingency table summarizing the relationship between two categorical variables.
     - When you need to understand the relationship between two categorical variables.
     - Contingency table showing frequencies of different categories.
   * - Log transformation
     - Transforms data to reduce skewness.
     - When you need to reduce the influence of extreme values and achieve a more normal-like distribution.
     - Transformed values of the dataset, typically reducing skewness.
   * - Hyperparameter Tuning
     - Process of finding the optimal parameters (hyperparameters) for a machine learning model
     - When you have a machine learning algorithm and need to improve its performance by tuning its hyperparameters
     - Best hyperparameters that maximize the model's performance metrics like accuracy or F1-score
   * - Model Evaluation and Selection
     - Assessing the performance of machine learning models and choosing the best one for your problem
     - After training multiple models, typically using cross-validation techniques to evaluate and compare their performance
     - Metrics such as accuracy, precision, recall, F1-score, confusion matrix, ROC curve
   * - Model Selection and Training
     - The process of choosing a suitable machine learning algorithm and training it on the dataset
     - At the beginning of a machine learning project when you have to select the most appropriate algorithm for your data
     - Trained machine learning model ready to make predictions on new data
   * - Feature Engineering
     - Process of creating new input features from existing ones or transforming existing features to improve model performance
     - Before training a machine learning model, to improve its predictive accuracy or to facilitate the learning process
     - A dataset with new features that may better represent the underlying patterns in the data
   * - ANOVA (f_oneway)
     - Analysis of Variance test used to compare the means of two or more groups
     - When comparing means of multiple groups to determine if there is a statistically significant difference between them
     - F-statistic and p-value indicating the significance of the differences between group means
   * - t-test (ttest_ind)
     - Independent samples t-test used to determine if there is a significant difference between the means of two groups
     - When comparing means of two groups to test whether they are significantly different from each other
     - t-statistic and p-value indicating the significance of the difference between group means
   * - Mann-Whitney U test (mannwhitneyu)
     - Non-parametric test used to determine if there is a significant difference between two independent groups
     - When the data does not meet the assumptions of parametric tests like the t-test
     - U-statistic and p-value indicating the significance of the difference between group distributions
   * - Kruskal-Wallis test (Kruskal)
     - Non-parametric test used to determine if there are significant differences between three or more independent groups
     - When comparing distributions of multiple groups to test whether they are significantly different from each other
     - H-statistic and p-value indicating the significance of the differences between group distributions
   * - Principal Component Analysis (PCA)
     - Dimensionality reduction technique that transforms the dataset into a lower-dimensional space
     - When dealing with high-dimensional data to reduce its dimensionality and visualize its structure
     - Reduced dataset with principal components that explain the maximum variance in the data
   * - t-distributed Stochastic Neighbor Embedding (TSNE)
     - Non-linear dimensionality reduction technique that is well-suited for embedding high-dimensional data for visualization
     - When visualizing high-dimensional data to understand the relationships between data points
     - Reduced dataset in lower-dimensional space that preserves the local structure of the data
   * - StandardScaler
     - Method for standardizing features by removing the mean and scaling to unit variance
     - Before training a machine learning model to ensure all features have the same scale and distribution
     - Transformed dataset with scaled features
   * - Accuracy Score
     - Metric for evaluating classification models, calculated as the ratio of correctly predicted instances to the total instances
     - After predicting outcomes with a classification model to evaluate its performance
     - Single numeric value representing the model's accuracy on the test dataset
   * - Silhouette Score
     - Metric for evaluating the quality of clusters formed by a clustering algorithm
     - After clustering data points to assess how well-separated the clusters are
     - Single numeric value indicating the quality of the clustering (ranging from -1 to 1)
   * - Machine Learning
     - A field of study that gives computers the ability to learn from data without being explicitly programmed.
     - When automating tasks that require pattern recognition.
     - Trained models, predictions, accuracy metrics.
   * - Classification
     - A supervised learning task where the goal is to predict the categorical class labels of new instances, based on past observations.
     - When the output variable is a category.
     - Class labels, confusion matrix, classification report.
   * - Decision Tree Classifier
     - A decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.
     - When you need an interpretable model that handles non-linear relationships well.
     - Decision tree structure, feature importances, classification metrics.
   * - Model Training
     - The process of feeding a machine learning algorithm with data to learn from, in order to make predictions.
     - When you have prepared data and are ready to create a predictive model.
     - Trained model, learned parameters.
   * - Model Evaluation
     - The process of assessing the performance of a trained model on new, unseen data.
     - After training a model to understand its performance.
     - Accuracy, precision, recall, F1-score, confusion matrix.
   * - Confusion Matrix
     - A table used to describe the performance of a classification model on a set of test data for which the true values are known.
     - To evaluate the performance of a classification model.
     - True positives, true negatives, false positives, false negatives.
   * - Classification Report
     - A report that includes the precision, recall, F1-score, and support for each class.
     - To get a detailed performance analysis of a classification model.
     - Precision, recall, F1-score, support for each class.
   * - K-Fold Cross-Validation
     - A resampling procedure used to evaluate machine learning models on a limited data sample.
     - When you want to validate the stability of your model.
     - Average performance metrics across folds.
   * - Hyperparameter Tuning
     - The process of optimizing the hyperparameters of a machine learning model to improve its performance.
     - When you need to improve model performance.
     - Best hyperparameters, improved performance metrics.
   * - Model Visualization
     - The process of graphically representing the structure and predictions of a model.
     - To understand or communicate the model's decisions.
     - Decision tree plots, feature importance plots.
   * - Data Exploration and Preprocessing
     - The initial steps of data analysis that involve summarizing and visualizing the main characteristics of a dataset, and preparing it for modeling.
     - Before model training to understand and clean data.
     - Summary statistics, visualizations, cleaned dataset.
   * - Handling Missing Values
     - Techniques used to handle missing data in a dataset, which can include deletion, imputation, or using algorithms that support missing values.
     - When your dataset contains missing values.
     - Imputed dataset, dataset with missing values handled.
   * - Data Cleaning
     - The process of detecting and correcting (or removing) corrupt or inaccurate records from a dataset.
     - Before model training to ensure data quality.
     - Cleaned dataset.
   * - Feature Engineering
     - The process of using domain knowledge to create features that make machine learning algorithms work better.
     - To improve model performance with better features.
     - New or transformed features.
   * - Exploratory Data Analysis (EDA)
     - An approach to analyzing datasets to summarize their main characteristics, often with visual methods.
     - To understand the dataset and uncover patterns.
     - Visualizations, summary statistics.
   * - Feature Scaling
     - The process of normalizing the range of independent variables or features of data.
     - When features have different scales and ranges.
     - Scaled feature values.
   * - K-means Clustering Algorithm
     - A method of vector quantization that partitions n observations into k clusters in which each observation belongs to the cluster with the nearest mean.
     - For clustering similar data points together.
     - Cluster labels for each data point.
   * - Elbow Method for Determining Optimal Clusters
     - A method to find the appropriate number of clusters in a dataset by fitting the model with a range of values for k and plotting the cost function.
     - To determine the optimal number of clusters in clustering.
     - Plot of cost function vs. number of clusters.
   * - Customer Profiling
     - The practice of categorizing customers based on common characteristics to better understand their needs and preferences.
     - For market segmentation and personalized marketing.
     - Customer segments, profiles.
   * - Visualization
     - The graphical representation of data to help understand and communicate insights.
     - To explore data or present findings.
     - Plots, charts, graphs.
   * - Saving Plots as PNG Files
     - The process of saving generated plots to a file in PNG format.
     - When you need to save visualizations for reports or presentations.
     - PNG image files.
   * - Scikit-learn
     - A free software machine learning library for the Python programming language.
     - For implementing machine learning algorithms.
     - Machine learning models, performance metrics.
   * - Breast Cancer Wisconsin Dataset
     - A popular dataset used for machine learning research and education, available from the UCI Machine Learning Repository.
     - For binary classification tasks.
     - Feature matrix, target vector.
   * - UCI Machine Learning Repository
     - A collection of databases, domain theories, and datasets used for empirical research in machine learning.
     - To find benchmark datasets for machine learning tasks.
     - Various datasets for different tasks.
   * - Classification Models
     - Algorithms used to predict the categorical class labels of new instances.
     - When the target variable is categorical.
     - Predicted class labels, performance metrics.
   * - Decision Tree 
     - A type of model used for classification and regression tasks that predicts the value of a target variable by learning simple decision rules from data features.
     - When an interpretable model is required.
     - Decision tree structure, predicted values.
   * - Classifier
     - An algorithm that maps input data to a specific category.
     - For tasks involving categorical outcomes.
     - Predicted class labels.
   * - Performance Metrics
     - Quantitative measures used to evaluate the performance of a model.
     - To assess model effectiveness.
     - Accuracy, precision, recall, F1-score.
   * - Precision
     - The ratio of true positive predictions to the total number of positive predictions made.
     - When the cost of false positives is high.
     - Precision score.
   * - Recall
     - The ratio of true positive predictions to the total number of actual positive instances.
     - When the cost of false negatives is high.
     - Recall score.
   * - F1-score
     - The harmonic mean of precision and recall.
     - When you need a balance between precision and recall.
     - F1-score.
   * - Loading Dataset
     - The process of importing a dataset into a Python environment for analysis.
     - As the first step in data analysis or modeling.
     - Dataframe or array containing the dataset.
   * - Converting to DataFrame
     - Transforming data into a pandas DataFrame, a 2-dimensional labeled data structure.
     - When working with data in pandas.
     - Pandas DataFrame.
   * - Checking Distribution of Target Labels
     - Analyzing the frequency of each class label in the target variable.
     - To understand class balance in classification tasks.
     - Frequency counts of target labels.
   * - Checking for Missing Values
     - Identifying any missing data in the dataset.
     - Before data preprocessing and cleaning.
     - Count of missing values per feature.
   * - Feature Scaling
     - The process of normalizing the range of independent variables or features of data.
     - When features have different scales and ranges.
     - Scaled feature values.
   * - Train-Test Split
     - The process of splitting a dataset into two parts: one for training the model and one for testing its performance.
     - To evaluate model performance on unseen data.
     - Training and testing datasets.
   * - Model Training
     - The process of feeding a machine learning algorithm with data to learn from, in order to make predictions.
     - When you have prepared data and are ready to create a predictive model.
     - Trained model, learned parameters.
   * - Model Evaluation
     - The process of assessing the performance of a trained model on new, unseen data.
     - After training a model to understand its performance.
     - Accuracy, precision, recall, F1-score, confusion matrix.
   * - Model Comparison
     - The process of comparing the performance of different machine learning models to choose the best one.
     - When selecting the best model for deployment.
     - Comparison metrics, model performance rankings.
   * - Model Recommendation
     - Providing suggestions on which model to use based on performance metrics and specific use-case requirements.
     - To select the most suitable model for a task.
     - Recommended model, rationale for selection.
