# Programming in Python

---

## Overview

This repository contains a comprehensive collection of Python programming concepts, techniques, and libraries applied to various tasks, including data analysis, machine learning, and more. It also contains different projects on Machine learning and statistical analysis. Each folder contains detailed information and codes for specific task (all developed using Jupyter Notebook) as well as a `.md` file to explain every aspect of each folder.

---

## Concepts and Techniques Covered (Including but not limited to)

All concepts and techniques covered can be found in the "Introduction, Python" folder, including details such as definition/descriptions, when to use each concept, typical output, definitions, and examples. More details on all concepts can be explored in the projects and other folders in more details with images. 

1. **Python Basics**:
   - Variables, Data types (integers, floats, strings)
   - Input/Output (printing, `input()` function)
   - Arithmetic Operations
   - String Manipulation
   - Conditional Statements
   - Comparison Operators
   - Basic Mathematical Operations

2. **Arithmetic Operations**:
   - Addition, subtraction, multiplication, division

3. **String Manipulation**:
   - String methods (`lower()` function)

4. **Conditional Statements**:
   - `if` statements

5. **User Input**:
   - `input()` function for user interaction

6. **Data Types**:
   - `float`, `str`

7. **Comparison Operators**:
   - `>=` (greater than or equal to)

8. **Basic Mathematical Operations**:
   - Performing calculations and printing results.

9. **Data Loading and Handling**:
   - Reading CSV files using pandas (`pd.read_csv()`).
   - Checking columns and dataset head (`df.columns`, `df.head()`).
   - Handling missing data and duplicates.

10. **Data Exploration**:
    - Exploring dataset properties (columns, shape, etc.).
    - Descriptive statistics (`describe()`, `info()`).
    - Correlation analysis (`corr()`).
    - Skewness and kurtosis (`skew()`, `kurt()`).

11. **Data Visualization**:
    - Matplotlib and Seaborn for plotting.
    - Various types of plots:
      - Scatter plots (`scatterplot()`).
      - Pair plots (`pairplot()`).
      - Histograms and density plots (`distplot()`, `histplot()`).
      - Box plots (`boxplot()`).
      - Violin plots (`violinplot()`).
      - Heatmaps (`heatmap()`).
      - Confusion matrix plots (`ConfusionMatrixDisplay`).
      - ROC curves (`roc_curve()`, `auc()`).

12. **Statistical Analysis**:
    - Hypothesis testing (`t-test`, `ANOVA`).
    - Probability distributions.
    - Confidence intervals.

13. **Machine Learning**:
    - **Classification**:
      - Logistic Regression (`LogisticRegression`).
      - Support Vector Machines (`SVC`).
      - K-Nearest Neighbors (`KNeighborsClassifier`).
      - Decision Trees (`DecisionTreeClassifier`).
      - Random Forests (`RandomForestClassifier`).
      - Multi-class classification.
      - Model evaluation metrics (accuracy, precision, recall, F1-score).
      - Confusion matrix (`confusion_matrix()`).
      - ROC curves (`roc_curve()`, `auc()`).

    - **Clustering**:
      - K-Means clustering (`KMeans`).
      - Hierarchical clustering.

    - **Dimensionality Reduction**:
      - Principal Component Analysis (PCA).
      - t-Distributed Stochastic Neighbor Embedding (t-SNE).

14. **Feature Engineering**:
    - Feature scaling (`StandardScaler`).
    - Feature selection techniques.
    - Handling categorical variables (encoding).

15. **Model Evaluation**:
    - Cross-validation (`cross_val_score()`).
    - Train-test split (`train_test_split()`).
    - Hyperparameter tuning.

16. **Time Series Analysis**:
    - Handling time series data.
    - Time series decomposition (`seasonal_decompose()`).
    - Forecasting techniques.

17. **Natural Language Processing (NLP)**:
    - Text preprocessing (tokenization, stemming, lemmatization).
    - Text vectorization (TF-IDF, CountVectorizer).
    - Sentiment analysis.
    - Topic modeling (LDA, NMF).

18. **Web Scraping**:
    - Using libraries like `requests`, `BeautifulSoup` for web scraping.

19. **Geospatial Analysis**:
    - Visualizing geospatial data using `folium`.
    - Handling geospatial data (shapefiles, GeoJSON).

20. **Deep Learning**:
    - Using libraries like TensorFlow and PyTorch.
    - Building neural networks (CNNs, RNNs).
    - Transfer learning.

21. **Reinforcement Learning**:
    - Implementing algorithms (Q-learning, SARSA).
    - OpenAI Gym.

22. **Model Deployment**:
    - Using Flask for web application deployment.
    - Containerization with Docker.
    - Cloud deployment (AWS, GCP, Azure).

---

## Libraries Used (Including but not limited to)

- **Data Handling and Analysis**:
  - Pandas (`import pandas as pd`).
  - NumPy (`import numpy as np`).
  - Scipy (`import scipy.stats as stats`).
  - yfinance (for collecting historical stock data).

- **Data Visualization**:
  - Matplotlib (`import matplotlib.pyplot as plt`).
  - Seaborn (`import seaborn as sns`).

- **Machine Learning**:
  - Scikit-learn (for machine learning models and tools, including preprocessing, model selection, and evaluation) - `from sklearn.model_selection import train_test_split, cross_val_score; from sklearn.metrics import ...`.
  - Statsmodels (`import statsmodels.api as sm`).
  - TensorFlow (`import tensorflow as tf`), PyTorch (`import torch`).
    
- **Natural Language Processing (NLP)**:
  - NLTK (`import nltk`).
  - SpaCy (`import spacy`).

- **Web Scraping**:
  - Requests (`import requests`).
  - BeautifulSoup (`from bs4 import BeautifulSoup`).

- **Geospatial Analysis**:
  - Folium (`import folium`).

- **Deep Learning**:
  - TensorFlow (`import tensorflow as tf`).
  - PyTorch (`import torch`).

- **Reinforcement Learning**:
  - Gym (`import gym`).

- **Cloud and Deployment**:
  - Flask (`from flask import Flask, ...`).
  - Docker (`Dockerfile`).
  - AWS, GCP, Azure.

- **Additional Libraries**:
  - `datetime` for handling date and time data.
  - `re` for regular expressions.
  - `itertools` for combinatorial iterators.
  - `pickle` for object serialization.
  - statsmodels (for exploring data, estimating statistical models, and performing statistical tests).

---

## Specific Tests and Tasks (Including but not limited to)

- Loading and handling dataset from a CSV file.
- Exploratory Data Analysis (EDA).
- Plotting various types of graphs and charts to visualize data distribution and relationships.
- Performing statistical analysis like skewness, kurtosis, and hypothesis testing.
- Applying machine learning algorithms for classification, clustering, and regression tasks.
- Evaluating model performance using various metrics and techniques.
- Creating visualizations like confusion matrices, ROC curves, and heatmaps.
- Implementing feature engineering techniques such as feature scaling, selection, and transformation.
- Applying time series analysis techniques such as decomposition and forecasting.
- Conducting web scraping using Python libraries.
- Stock Market Analysis
- Customer segmentation for E-commerce
- Performing geospatial analysis and visualization.
- Handling natural language processing tasks like text preprocessing, vectorization, sentiment analysis, and topic modeling.
- Data preprocessing tasks like outlier handling, imputation, and normalization.
- Model evaluation and selection using cross-validation and hyperparameter tuning.
- Implementing deep learning models using TensorFlow and PyTorch.
- Reinforcement learning tasks with OpenAI Gym.
- Deploying models using Flask and Docker, and deploying to cloud platforms like AWS, GCP, and Azure.

---

## Others

- **Python Skills**:
  - Understanding of Python syntax and structure.
  - Ability to handle data using pandas and numpy.
  - Familiarity with plotting libraries like Matplotlib and Seaborn.
  - Practical experience in machine learning and data analysis tasks.
  - Experience with web scraping, geospatial analysis, and natural language processing.
  - Familiarity with deep learning frameworks like TensorFlow and PyTorch.
  - Experience with reinforcement learning using Gym.
  - Knowledge of cloud deployment and containerization using Docker.
  - Familiarity with Flask for web application deployment.

- **Advanced Python Concepts Covered**:
  - Loops (for, while)
  - Lists, tuples, dictionaries
  - Functions

- **Error Handling**:
  - Try-except blocks for handling exceptions.

- **Modules and Packages**:
  - Importing and using built-in modules (`math`, `random`, etc.)
    
- **File Handling**:
  - Reading from and writing to files.

- **Object-Oriented Programming (OOP)**:
  - Classes and objects.

---

## Images
A lot of images are availabe in the images folder to refer back to the results acieved throughout various analysis. 

---

## Machine Learning and Statistical Analysis Portfolio: Iris Dataset

#### Project 1: Species Classification Using Machine Learning
Classifying Iris flower species with Logistic Regression, SVM, Decision Trees, and Random Forest; learned model evaluation and hyperparameter tuning.

#### Project 2: Exploratory Data Analysis (EDA) on the Iris Dataset
Understanding Iris dataset through statistical analysis and visualizations; learned data exploration and hypothesis testing.

#### Project 3: Dimensionality Reduction Analysis
Reducing dimensionality of Iris dataset using PCA and t-SNE; learned feature scaling, dimensionality reduction, and data visualization.

#### Project 4: Comparative Analysis of Iris Species
Comparing Iris species using statistical tests and visualizations; learned statistical testing, effect size calculation, and data visualization.

#### Project 5: Cluster Analysis on the Iris Dataset
Grouping Iris flowers into clusters using K-means clustering; learned clustering, feature preprocessing, and evaluation metrics.

---

## Scikit-learn Projects

#### Project 1: Customer Segmentation for E-commerce

**Description:** Conducted customer segmentation using K-means clustering based on purchasing behavior in an e-commerce dataset.

**Key Objectives:** Identified distinct customer segments to tailor marketing strategies and improve business decision-making.

**Key Steps:**
- **Data Exploration and Preprocessing:** Ensured data quality by handling missing values, duplicates, and formatting dates. Engineered features like TotalAmount for customer spending quantification.
- **Customer Segmentation using Clustering:** Implemented K-means clustering to group customers with similar purchasing patterns. Determined optimal clusters using the elbow method and visualized segment distributions.
- **Customer Profiling and Insights:** Analyzed segment characteristics, behavior patterns, and visualized metrics to derive actionable marketing insights.

**Overall Result:**
- Successfully segmented customers into meaningful groups based on their purchasing behavior, enabling personalized marketing strategies and business optimizations.
- Demonstrated proficiency in data preprocessing, clustering techniques, and deriving actionable insights from customer data in an e-commerce context.

---

#### Project 2: AAPL Stock Price Prediction with Machine Learning

**Description:** Built a machine learning model to forecast Apple Inc. (AAPL) stock prices for the next 30 days using historical data.

**Key Objectives:** Predicted future stock prices to provide insights for investment decisions and understand potential price trends.

**Key Steps:**
- **Data Collection and Preprocessing:** Gathered historical stock data using yfinance, cleaned, and engineered features like moving averages and Relative Strength Index (RSI).
- **Model Selection and Training:** Trained models including Linear Regression, RandomForestRegressor, and SVR. Evaluated model performance based on metrics like RMSE and R-squared.
- **Prediction and Evaluation:** Forecasted AAPL stock prices for January 2023 and visualized actual vs. predicted results to assess model accuracy.

**Overall Result:**
- Successfully developed a predictive model that effectively forecasted AAPL stock prices, demonstrating the application of machine learning in financial forecasting.
- Gained practical experience in data preprocessing, feature engineering, model selection, and evaluation for stock price prediction.

--- 

## Skills Learned
- Machine learning model selection, evaluation, and hyperparameter tuning.
- Data exploration, visualization techniques, and hypothesis testing.
- Dimensionality reduction using PCA and t-SNE.
- Statistical analysis including ANOVA, t-tests, and effect size calculations.
- Unsupervised learning techniques and cluster analysis using K-means.
- Python programming skills with pandas, scikit-learn, and numpy.
- Time Series Analysis: Ability to analyze and model temporal data, including trend analysis, seasonality decomposition, and forecasting techniques such as ARIMA and Prophet.
- Feature Engineering: Proficiency in creating new features from existing data to enhance predictive model performance, including domain-specific feature selection and transformation techniques.
- Deployment and Productionization: Understanding of deploying machine learning models into production environments, including containerization, API development (e.g., Flask or FastAPI), and model monitoring for performance and drift detection.

--- 

## Explore Each Project
Navigate to each project folder for detailed code, results, and further documentation.

--- 

## Summary of What Did I Learn?

- Throughout this project, I expanded my knowledge across various domains of Python programming and data science. 
- I gained proficiency in data handling, exploratory data analysis, machine learning, natural language processing, deep learning, and reinforcement learning. 
- I learned to effectively use libraries such as pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, and PyTorch and many other libraries.
- Acquired skills in cross-validation techniques to assess model generalization and performance metrics selection to measure model efficacy.
- Additionally, I gained experience in web scraping, geospatial analysis, and model deployment using Flask and Docker. 
- This respository helped me strengthen my skills in Python programming, data analysis, machine learning model development and much more.

---

## What I Enjoyed the Most Overall

- The most enjoyable aspect of this project was the application of machine learning algorithms and deep learning models.
- I particularly liked experimenting with different classification and clustering techniques to solve real-world problems.
- Building and optimizing neural networks using TensorFlow and PyTorch was both challenging and rewarding.
- Working on predictive models to forecast trends and make informed decisions based on data insights was highly satisfying.
- Moreover, the process of deploying models using Flask and Docker, and integrating them with cloud platforms, was another aspect that I found fascinating.
  
---

## Conclusion

In conclusion, this respository provided me with a comprehensive hands-on experience in Python programming and data science. I gained valuable insights into various machine learning, statistical analysis and deep learning techniques, as well as practical skills in data visualization, natural language processing, and geospatial analysis. This project not only reinforced my understanding of core concepts but also equipped me with the skills necessary to tackle complex data-driven challenges. Moving forward, I plan to further enhance this project by integrating advanced techniques and exploring new areas of interest in data science.

---

## Usage

### Setup
1. Clone the repository:
 ```bash
git clone https://github.com/your_username/project_name.git
cd project_name
```
2. Install the required dependencies:
```bash
Copy code
pip install -r requirements.txt
```
3. Running Notebooks
Navigate to the notebooks directory:
```
cd notebooks
```
4. Start Jupyter Notebook:
```
jupyter notebook
```
5. Open and run the desired notebook.

---

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

**Please follow these steps to contribute to the project:** 
1. Fork the repository and clone it to your local machine.
2. Create a new branch: `git checkout -b feature/my-new-feature`.
3. Make your changes and test them thoroughly.
4. Commit your changes: `git commit -am 'Add some feature'`.
5. Push to the branch: `git push origin feature/my-new-feature`.
6. Submit a pull request.
7. I will review your changes and work with you to integrate them into the project.

---

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, provided that appropriate credit is given to the original author.
