# AAPL Stock Price Prediction with Machine Learning

## Description
This project focuses on building a predictive model to forecast the closing prices of Apple Inc. (AAPL) stock for the next 30 days using historical data. The model aims to provide insights into potential future price trends based on past performance.

## Aim
To build a predictive machine learning model that accurately forecasts the closing prices of Apple's stock (AAPL) for the next 30 days (02-31/01/2023) using 5 years of historical data from January 1, 2018, to January 1, 2023. The model will be evaluated based on its accuracy and effectiveness in predicting future stock prices.

## Need for the Project
The project aims to:
- Provide insights into the potential future performance of AAPL stock.
- Demonstrate the application of machine learning in financial forecasting.
- Explore the accuracy of predictive models in capturing stock price movements over time.

## Steps Overview
1. **Data Collection:** (Completed) Obtain historical stock data using `yfinance`.
2. **Data Preprocessing:** Clean and preprocess the data using various techniques.
3. **Exploratory Data Analysis (EDA):** Visualize and understand the data.
4. **Feature Engineering and Selection:** Create and select relevant features for the model.
5. **Model Selection, Training and Prediction:** Train different models and select the best one to make a prediction .
6. **Prediction and Visualization:** Finalise the predictions and visualize the results.

## Detailed Steps Involved and Why They Were Needed
1. **Data Collection**: Historical stock data was collected using `yfinance` to gather necessary data for training and evaluation.
   
2. **Data Preprocessing**: Cleaning, filtering, and transforming data ensure that the dataset is suitable for modeling. This step included handling missing values, calculating additional features like moving averages and RSI, and ensuring the data is in a format suitable for analysis.

```python
# Feature Engineering (example: adding moving averages)
stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()

# Create lag features for closing price
stock_data['Close_Lag1'] = stock_data['Close'].shift(1)
stock_data['Close_Lag2'] = stock_data['Close'].shift(2)
stock_data['Close_Lag3'] = stock_data['Close'].shift(3)

# Relative Strength Index (RSI) calculation
def calculate_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

stock_data['RSI'] = calculate_RSI(stock_data)
```   
**Objective of the step:** Enhance the dataset with additional features that could improve the model’s predictive power and select the most relevant features for the model.

```python
In this step, we enhanced the dataset by creating additional features that could improve the predictive power of our model. Specifically:
1.	Lag Features: Added previous days' closing prices (Close_Lag1, Close_Lag2, Close_Lag3) to capture momentum and recent trends.
2.	Moving Averages: Calculated 50-day and 200-day moving averages (MA_50, MA_200) to reflect short-term and long-term trends.
3.	Relative Strength Index (RSI): Added the RSI to indicate overbought or oversold conditions in the stock.
```

3. **Exploratory Data Analysis (EDA)**: Visualization and statistical analysis of the data helped in understanding patterns, trends, and relationships within the dataset. This step was crucial for identifying potential features for the model and gaining insights into the stock's historical behavior.

| **Identified Trends and Patterns in AAPL Stock Prices**                                                                 |
|---------------------------------------------------------------------------------------------------------------------------|
| Plotting the daily closing prices as well as monthly averages can help in understanding how AAPL's stock prices behave over different time frames.                                                                                                       |

| **Aspect**                                                                                                               | **Summary**                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Overall Insight**                                                                                                      | AAPL's stock has experienced varying trends and cycles over the 5-year period.                                           |
| **2019-2020**                                                                                                            | Upward trend with some stability before.                                                                               |
| **Mid-2020**                                                                                                             | Significant increase followed by an uptrend towards the end of the year.                                               |
| **2021**                                                                                                                 | Continued upward movement with minor drops, reaching the second-highest peak observed.                                |
| **2022**                                                                                                                 | Highest peak reached at the start of 2022 but decline starting mid-year, with a notable drop, followed by a partial recovery towards the end. |
| **2023**                                                                                                                 | Starting with a decline continuing from 2022.                                                                          |
| **Long-term Trends**                                                                                                     | Generally upward from 2019 to mid-2022, with significant increases in mid-2020 and 2021.                              |
| **Cyclical Patterns**                                                                                                    | Periods of growth followed by corrections, indicating typical market cycles.                                           |
| **Recent Trends**                                                                                                        | A decline starting in mid-2022, with fluctuations in early 2023.                                                       |
| **Interpretation**                                                                                                       | The trends indicate periods of growth, consolidation, and correction in AAPL's stock prices. Understanding these patterns helps in predicting potential future movements. These trends suggest that AAPL's stock is subject to both long-term growth and shorter-term market corrections, influenced by broader market conditions and company-specific factors. |


| **Volume and Liquidity Analysis**                                                                                         |
|---------------------------------------------------------------------------------------------------------------------------|
| Analyzed trading volume over the same 5-year period to understand liquidity and market activity. High trading volumes often coincide with significant price movements.                                                                                                                                                      |

| **Aspect**                                                                                                               | **Summary**                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Overall Insight**                                                                                                      | Trading volume reflects changes in market activity and investor sentiment.                                             |
| **High Points**                                                                                                          | Peaks in start and mid-2020 coincided with significant price movements, indicating heightened investor interest and volatility (peaking slightly above 4.0). |
| **Declining Trend**                                                                                                      | Decrease in volume towards late 2022 and early 2023 (around 0.7 to 0.8) suggests reduced trading activity and possibly lower market volatility. |
| **Interpretation**                                                                                                       | High trading volumes often coincide with significant price movements, suggesting increased market activity during volatile periods like mid-2020. Volume analysis underscores the correlation between market liquidity, investor participation, and stock price movements, influencing trading strategies and market timing decisions. |

| **Correlation Analysis**                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------|
| Examined correlations between closing prices and volume. This helps in identifying which factors are most influential in predicting AAPL's stock prices.|

| **Aspect**                                                                                                               | **Summary**                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Overall Insight**                                                                                                      | Strong positive correlation between AAPL's closing prices and trading volume.                                         |
| **Correlation Heatmap**                                                                                                  | The heatmap showed a strong correlation between closing prices and volume, indicated by the red color. A correlation coefficient of 1 suggests a perfect positive correlation between these variables. |
| **High Correlation Coefficient**                                                                                         | Indicates that changes in trading volume are closely linked with changes in stock prices.                             |
| **Implication**                                                                                                          | Volume serves as a leading indicator for price movements, highlighting its predictive value in stock price forecasting. |
| **Interpretation**                                                                                                       | Understanding this correlation helps in identifying periods of market strength or weakness based on trading activity, supporting effective decision-making in trading and investment strategies. Volume is likely a significant factor influencing AAPL's stock prices, with higher volumes generally aligning with price movements. |


| **Time Series Decomposition**                                                                                             |
|---------------------------------------------------------------------------------------------------------------------------|
| Decomposed the time series into trend, seasonal, and residual components to identify seasonal fluctuations and overall trends, which can inform the seasonality component in your forecasting model.                                                                                                                                                      |

| **Aspect**                                                                                                               | **Summary**                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Overall Insight**                                                                                                      | Decomposition into trend, seasonal, and residual components reveals underlying dynamics of AAPL's stock prices.         |
| **Trend**                                                                                                                | Shows overall direction of AAPL's stock prices, highlighting periods of growth and decline. Started slow, picked up speed, peaked by mid-2022, and began to decline afterward. |
| **Seasonal Component**                                                                                                   | Indicates irregular fluctuations without a clear repetitive pattern, suggesting unpredictable market dynamics.         |
| **Residual**                                                                                                             | Represents random noise around the trend and seasonal components, capturing unexplained variability.                   |
| **Implication**                                                                                                          | Analyzing these components provides insights into market cycles, seasonal effects, and potential future price movements, supporting more accurate forecasting and risk management strategies. Seasonal patterns and overall trends identified through decomposition can help in understanding the cyclical nature of AAPL's stock prices, providing insights into potential future movements. |


| **Key Insights from EDA to Consider**                                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------|
| 1. **Trend and Cyclical Patterns:** Long-term growth and market corrections.                                               |
| 2. **Volume and Liquidity:** High correlation between trading volume and price movements.                                  |
| 3. **Seasonality:** Seasonal fluctuations and overall trends identified in the decomposition.                              |
| 4. **Correlation:** Strong correlation between volume and closing prices.                                                  |


4. **Feature Engineering and Selection**: Creating relevant features such as lagged values which helps to improves the model's ability to capture meaningful patterns from the data.

5. **Model Selection, Training, and Prediction**: Various machine learning models such as Linear Regression, RandomForestRegressor, and SVR were trained and evaluated to determine the best-performing model. Training involved using historical data, while prediction focused on forecasting AAPL stock prices for January 2023.

```python
The results show that all models performed exceptionally well, with the Linear Regression model having the highest R² score and the lowest RMSE, indicating it might be the most accurate for our data.
```

6. **Final Evaluation and Visualization**: The performance of the selected model was evaluated using metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared. Visualizations comparing actual vs. predicted stock prices for 01-31 January 2023 provided insights into model accuracy and performance.

## Results
The project successfully:
- Developed a predictive model that forecasts AAPL stock prices for January 2023.
- Evaluated model performance using standard regression metrics and visual comparisons.
- Demonstrated the feasibility of using machine learning for stock price prediction.

## Conclusion
Through this project, the goal of forecasting AAPL stock prices using machine learning techniques was achieved. The model showed promising results in predicting future price trends based on historical data that they are trend to increase in a similar manner as shown previosuly with the data. There will not be a significant difference between actual and predict prices but a minor. The insights gained contribute to understanding the dynamics of stock price movements and the applicability of predictive analytics in finance.

## Discussion
The project highlighted the importance of data preprocessing, feature engineering, and model selection in building effective predictive models for financial forecasting. The model was trained on historical data and validated against known outcomes, and its purpose is to predict stock prices for the next 30 days into the future. This forecasting horizon aligns with the common practice of short-term predictions in financial markets, aiming to provide insights into potential price movements over the specified period. Challenges such as data quality, model interpretability, and performance evaluation were addressed throughout the project.

## Usefulness and Future Implications
The developed model can be useful for:
- Investors seeking insights into future stock price movements.
- Financial analysts looking to enhance their forecasting capabilities.
- Academic research in the field of machine learning and finance.

Future implications include:
- Refining the model with additional data sources and advanced techniques.
- Incorporating real-time data for more accurate predictions.
- Extending the approach to other stocks and financial instruments.

## What Did I Learn
Through this project, I gained practical experience in:
- Collecting and preprocessing financial data for analysis.
- Applying machine learning algorithms to real-world financial forecasting.
- Evaluating and interpreting model performance metrics.
- Communicating insights and findings effectively through data visualization and documentation.
