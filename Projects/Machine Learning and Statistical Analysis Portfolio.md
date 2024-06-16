# Machine Learning and Statistical Analysis Portfolio: Iris Dataset

## Overview
This repository contains a series of projects focused on the application of machine learning and statistical techniques to the Iris dataset. Each project demonstrates a different aspect of machine learning and different types of statistical analysis, ranging from species classification, exploratory data analysis, dimensionality reduction, comparative analysis, and clustering. The Iris dataset is used throughout these projects to illustrate various concepts and methods.

## Projects

### Project 1: Species Classification Using Machine Learning

#### Description
This project focuses on classifying Iris flower species using various machine learning algorithms. The aim is to build and compare models to predict the species based on sepal and petal measurements.

#### Aim
To compare different machine learning algorithms and select the best-performing model for classifying Iris flower species.

#### Key Steps
- **Data Preprocessing**: Loading, exploring, and splitting the dataset; feature scaling.
- **Feature Engineering**: Creating new features and transforming existing ones.
- **Model Selection**: Initializing, training, and evaluating models (Logistic Regression, SVM, Decision Tree, Random Forest).
- **Model Evaluation**: Using accuracy, precision, recall, F1-score, and confusion matrices.
- **Hyperparameter Tuning**: Optimizing models using Grid Search Cross-Validation.

#### Results
The Support Vector Machine (SVM) model achieved the highest accuracy (100%) on the test set.

#### Learnings
- Data preprocessing and feature engineering.
- Model selection and evaluation techniques.
- Hyperparameter tuning with Grid Search CV.

---

### Project 2: Exploratory Data Analysis (EDA) on the Iris Dataset

#### Description
This project performs an Exploratory Data Analysis (EDA) on the Iris dataset to understand its structure through statistical and visualization techniques.

#### Aim
To understand the underlying structure of the Iris dataset, identify relationships between features, and distinguish between different species.

#### Key Steps
- **Data Summarization**: Loading data, descriptive statistics, correlation analysis.
- **Data Visualization**: Histograms, box plots, pair plots, violin plots.
- **Hypothesis Testing**: ANOVA tests to check for significant differences in features across species.

#### Results
The EDA revealed significant differences in petal measurements across species, with clear separations in scatter plots.

#### Learnings
- Importance of data summarization and visualization.
- Interpretation of statistical measures and hypothesis testing.

---

### Project 3: Dimensionality Reduction Analysis

#### Description
This project applies PCA and t-SNE to reduce the dimensionality of the Iris dataset and visualize it in two dimensions.

#### Aim
To employ PCA and t-SNE to visualize the dataset in lower dimensions, aiding in understanding species separability.

#### Key Steps
- **Feature Scaling**: Standardizing features.
- **Dimensionality Reduction**: Applying PCA and t-SNE.
- **Visualization**: Plotting reduced dimensions to observe clusters.

#### Results
Both PCA and t-SNE visualizations revealed distinct clusters corresponding to Iris species.

#### Learnings
- Importance of feature scaling.
- Application and strengths of PCA and t-SNE in visualizing high-dimensional data.

---

### Project 4: Comparative Analysis of Iris Species

#### Description
This project involves a comparative analysis of Iris species using statistical tests, visualizations, and effect size calculations.

#### Aim
To identify significant differences in morphological characteristics between Iris species.

#### Key Steps
- **Statistical Tests**: ANOVA and t-tests for sepal and petal dimensions.
- **Visualizations**: Box plots, violin plots, and swarm plots for data distribution.
- **Effect Sizes**: Calculating Cohen's d to measure the magnitude of differences.

#### Results
ANOVA and t-tests indicated significant differences in certain features, with visualizations supporting these findings.

#### Learnings
- Statistical testing and interpretation.
- Visualization techniques for comparative analysis.
- Calculating and interpreting effect sizes.

---

### Project 5: Cluster Analysis on the Iris Dataset

#### Description
This project applies K-means clustering to group Iris flowers into clusters based on their measurements.

#### Aim
To identify natural groupings within the Iris dataset using unsupervised learning techniques.

#### Key Steps
- **Feature Selection and Preprocessing**: Selecting numeric features, standardizing, and applying PCA.
- **Choosing Number of Clusters**: Using the Elbow method.
- **Applying K-means Clustering**: Assigning cluster labels and evaluating with Silhouette score.
- **Visualization**: Visualizing clusters in PCA-reduced space.

#### Results
K-means clustering revealed distinct clusters, with good quality indicated by the Silhouette score.

#### Learnings
- Application of K-means clustering and the Elbow method.
- Importance of feature scaling and dimensionality reduction.
- Evaluating and visualizing cluster quality.

---

## Conclusion
These projects collectively demonstrate various machine learning techniques applied to the Iris dataset. From classification and EDA to dimensionality reduction, comparative analysis, and clustering, each project provides valuable insights and practical skills in handling and analyzing data.

## What I Learned
- Comprehensive understanding of machine learning concepts.
- Proficiency in using Python libraries like pandas, scikit-learn, and numpy.
- Techniques for data preprocessing, feature engineering, model evaluation, and visualization.
- Practical experience in statistical analysis and unsupervised learning methods.
