# Data Science and Data Analysis Projects (Scikit-learn)

## Overview
This folder contains two projects that demonstrate the application of data science and data analysis techniques in financial forecasting and customer segmentation. The projects aim to provide valuable insights for business decision-making and enhance predictive capabilities using historical data.

## Projects

### Project 1: Customer Segmentation for E-commerce

**Description**
This project focuses on performing customer segmentation for an e-commerce dataset using data analysis and machine learning techniques. By clustering customers based on their purchasing behaviour, the aim is to identify distinct customer segments that can be targeted with personalized marketing strategies, ultimately improving customer engagement and sales.

**Aim**
The aim of this project is to segment customers into meaningful groups based on their purchasing behaviour using clustering algorithms. This segmentation will enable personalized marketing strategies and enhance business decision-making.

**Key Steps**
1. **Data Exploration and Preprocessing**: 
    - Data Loading
    - Handling Missing Values
    - Handling Duplicates
    - Date Formatting
    - Feature Engineering
    - Data Distribution Analysis
2. **Customer Segmentation using Clustering**:
    - Feature Selection
    - Feature Standardization
    - Determining Optimal Clusters using the Elbow Method
    - Clustering Execution
    - Cluster Visualization
3. **Customer Profiling and Insights**:
    - Cluster Statistics
    - Segment Characteristics
    - Behavior Patterns
    - Visualizations

**Results**
- Successfully segmented customers based on their purchasing behaviour using K-means clustering.
- Identified distinct customer segments with unique purchasing patterns.
- Provided actionable insights for targeted marketing and strategic decision-making.
- Created visual summaries illustrating key metrics per cluster.

**Learnings**
- Gained proficiency in advanced data preprocessing and feature engineering.
- Applied K-means clustering for customer segmentation.
- Derived actionable insights from data analysis and visualizations.
- Understood the importance of handling outliers and selecting appropriate features for clustering.

### Project 2: AAPL Stock Price Prediction with Machine Learning

**Description**
This project focuses on building a predictive model to forecast the closing prices of Apple Inc. (AAPL) stock for the next 30 days using historical data. The model aims to provide insights into potential future price trends based on past performance, combining data analysis and machine learning techniques.

**Aim**
To build a predictive machine learning model that accurately forecasts the closing prices of Apple's stock (AAPL) for the next 30 days (02-31/01/2023) using 5 years of historical data from January 1, 2018, to January 1, 2023. The model will be evaluated based on its accuracy and effectiveness in predicting future stock prices.

**Key Steps**
1. **Data Collection**: 
    - Obtained historical stock data using yfinance.
2. **Data Preprocessing**:
    - Cleaned and filtered data
    - Calculated additional features (e.g., moving averages, RSI)
    - Handled missing values and transformed data.
3. **Feature Engineering and Selection**:
    - Created lag features for closing prices
    - Calculated moving averages and RSI
4. **Exploratory Data Analysis (EDA)**:
    - Visualized data to understand patterns and trends
    - Conducted volume and liquidity analysis
    - Performed correlation analysis
    - Decomposed time series into trend, seasonal, and residual components
5. **Model Selection, Training, and Prediction**:
    - Trained various machine learning models (Linear Regression, RandomForestRegressor, SVR)
    - Selected the best-performing model
6. **Prediction and Visualization**:
    - Made predictions for January 2023
    - Visualized actual vs. predicted stock prices

**Results**
- Developed a predictive model that forecasts AAPL stock prices for January 2023.
- Evaluated model performance using standard regression metrics and visual comparisons.
- Demonstrated the feasibility of using machine learning for stock price prediction.
- Found that the Linear Regression model performed best in terms of R² score and RMSE.

**Learnings**
- Gained practical experience in collecting and preprocessing financial data.
- Applied machine learning algorithms to financial forecasting.
- Evaluated and interpreted model performance metrics.
- Communicated insights and findings through data visualization and documentation.

## Conclusion
Both projects successfully achieved their respective aims, demonstrating the application of data science and data analysis techniques in different domains. The customer segmentation project provided valuable insights for targeted marketing in e-commerce, while the stock price prediction project showcased the potential of predictive analytics in financial forecasting. These projects highlight the importance of data preprocessing, feature engineering, and model selection in building effective models.

## What I Learned
Through these projects, I gained comprehensive experience in:
- Data collection, preprocessing, and feature engineering.
- Applying clustering and regression algorithms to real-world problems.
- Analyzing and visualizing data to derive actionable insights.
- Evaluating model performance and communicating results effectively.
- Addressing challenges such as data quality, outliers, and feature selection in both data analysis and data science projects.
