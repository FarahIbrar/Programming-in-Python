#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction

# ### Step 1: Data Collection

# In[1]:


# Install yfinance to collect the historical stock data
get_ipython().system('pip install yfinance')


# In[133]:


import yfinance as yf
import pandas as pd
import warnings


# In[134]:


# Suppress warnings
warnings.filterwarnings("ignore")


# In[135]:


# Define the stock ticker and the time period
ticker = 'AAPL'  # You can change this to any stock ticker you're interested in. This ticker symbol is for Apple Inc.
start_date = '2018-01-01'
end_date = '2023-01-31'


# In[136]:


# Download historical stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
stock_data.to_csv('historical_stock_data.csv')


# In[137]:


print("Data collection complete. The data has been saved to 'historical_stock_data.csv'.")


# In[ ]:





# ### Step 2: Data Preprocessing

# In[138]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[139]:


# Load the data from the CSV file
stock_data = pd.read_csv('historical_stock_data.csv')


# In[140]:


# Check for missing values
missing_values = stock_data.isnull().sum()
print("Missing values:\n", missing_values)


# In[141]:


stock_data['Date'] = pd.to_datetime(stock_data['Date'])


# In[142]:


# Feature Engineering (example: adding moving averages)
stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()


# In[143]:


# Create lag features for closing price
stock_data['Close_Lag1'] = stock_data['Close'].shift(1)
stock_data['Close_Lag2'] = stock_data['Close'].shift(2)
stock_data['Close_Lag3'] = stock_data['Close'].shift(3)


# In[144]:


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


# In[145]:


# Drop initial rows with NaN values from lag and RSI calculations
stock_data.dropna(inplace=True)


# In[154]:


print("Data preprocessing complete.")


# In[ ]:





# ### Step 3: Exploratory Data Analysis (EDA)

# In[146]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[147]:


# Plotting daily historical closing prices
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Date'], stock_data['Close'], marker='o', linestyle='-', color='b', label='Daily Closing Price')
plt.title('Historical Closing Prices of AAPL')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('closing_prices_daily.png')  # Save the plot
plt.show()

# Aggregating to monthly data for clearer trend analysis
stock_data['Date'] = pd.to_datetime(stock_data['Date'])  # Ensure 'Date' column is datetime type
stock_data.set_index('Date', inplace=True)
stock_data_monthly = stock_data.resample('M').mean().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(stock_data_monthly['Date'], stock_data_monthly['Close'], marker='o', linestyle='-', color='b', label='Monthly Avg Closing Price')
plt.title('Monthly Average Closing Prices of AAPL')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('closing_prices_monthly.png')  # Save the plot
plt.show()


# In[148]:


# Plotting trading volume for Volume analysis

plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, stock_data['Volume'], marker='o', linestyle='-', color='g', label='Trading Volume')
plt.title('Trading Volume of AAPL')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('trading_volume.png')  # Save the plot
plt.show()


# In[149]:


### Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')  # Save the plot
plt.show()


# In[150]:


### Time Series Decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposing time series
result = seasonal_decompose(stock_data['Close'], model='multiplicative', period=90)  # Adjust period as needed (quarterly patterns )
result.plot()
plt.tight_layout()
plt.savefig('time_series_decomposition.png')  # Save the plot
plt.show()


# In[ ]:





# ### Step 4: Feature Engineering and Selection

# In[155]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[156]:


# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Volume', 'MA_50', 'MA_200', 'RSI']
target = 'Close'
X = stock_data[features]
y = stock_data[target]


# In[157]:


# Normalize/Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:





# ### Step 5: Model Selection, Training and prediction

# In[186]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings


# In[159]:


# Split the data ensuring that the test set includes January 2023
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# In[175]:


# Train the model
final_model_lr = LinearRegression()
final_model_lr.fit(X_train, y_train)

# Save the model
joblib.dump(final_model_lr, 'final_linear_regression_model.pkl')


# In[180]:


# Predict
y_pred = final_model_lr.predict(X_test)


# In[177]:


# Print results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("\n")


# In[ ]:





# ### Step 6: Final Evaluation and Visualization

# In[178]:


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[183]:


# Step 1: Reset the index to make 'Date' a column
stock_data_reset = stock_data.reset_index()

# Step 2: Ensure the 'Date' column is in datetime format
stock_data_reset['Date'] = pd.to_datetime(stock_data_reset['Date'])

# Step 3: Slice the data from train_size onwards
test_slice = stock_data_reset.iloc[train_size:]

# Step 4: Extract and filter the relevant dates
test_dates = test_slice['Date'].reset_index(drop=True)
january_2023_dates = test_dates[(test_dates >= '2023-01-02') & (test_dates <= '2023-01-31')]

print(january_2023_dates)


# In[184]:


# Align indices for January 2023
january_indices = test_dates[(test_dates >= '2023-01-02') & (test_dates <= '2023-01-31')].index


# In[185]:


# Filter y_test and y_pred for January 2023 using aligned indices
y_test_january_2023 = y_test.iloc[january_indices]
y_pred_january_2023 = y_pred[january_indices]


# In[187]:


# Plotting Actual vs Predicted Prices for January 2nd to January 31st, 2023
plt.figure(figsize=(14, 7))
plt.plot(january_2023_dates, y_test_january_2023, marker='o', linestyle='-', color='b', label='Actual Price')
plt.plot(january_2023_dates, y_pred_january_2023, marker='o', linestyle='-', color='r', label='Predicted Price')
plt.title('Actual vs Predicted Stock Prices (January 2nd - January 31st, 2023)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_prices_january_2023.png')  # Save the plot
plt.show()


# In[188]:


##### Print the summary of key findings and results
print("Final Linear Regression Model Summary:")
print(f"Best Parameters: {best_params_lr}")
print(f"Final RMSE: {final_rmse}")
print(f"Final MAE: {final_mae}")
print(f"Final R2 Score: {final_r2}")
print(f"Cross-Validation RMSE: {mean_cv_rmse}")

print("The final model has been saved and key results have been summarized.")

