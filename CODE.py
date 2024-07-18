# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:24:29 2024

@author: pidav
"""

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# List of ETFs
etfs = ['SUSW.MI', 'IESE.AS', 'ELCR.PA', 'GRON.F', 'ICLN', 'AYEM.DE', 'DJRE.AX']

# Fetching data from Yahoo Finance
data = yf.download(etfs, start="2020-01-01", end="2024-05-05", group_by='ticker')

# Clean data with backward and forward propagation
cleaned_data = {}
for etf in etfs:
    df = data[etf]['Adj Close'].copy()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    cleaned_data[etf] = df

# Plot adjusted close prices for each ETF
plt.figure(figsize=(14, 8))
for etf in etfs:
    plt.plot(cleaned_data[etf], label=etf)
plt.title('Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Function to plot moving averages
def plot_moving_averages(df, title, window1=100, window2=250):
    ma100 = df.rolling(window=window1).mean()
    ma250 = df.rolling(window=window2).mean()
    
    plt.figure(figsize=(14, 8))
    plt.plot(df, label='Adjusted Close')
    plt.plot(ma100, label=f'{window1}-day MA')
    plt.plot(ma250, label=f'{window2}-day MA')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Plot moving averages for each ETF
for etf in etfs:
    plot_moving_averages(cleaned_data[etf], f'{etf} Moving Averages')

# Function to calculate weighted moving average and plot
def weighted_moving_average(df, window):
    weights = np.arange(1, window + 1)
    wma = df.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return wma

def plot_weighted_moving_average(df, title, window=100):
    wma = weighted_moving_average(df, window)
    
    plt.figure(figsize=(14, 8))
    plt.plot(df, label='Adjusted Close')
    plt.plot(wma, label=f'{window}-day WMA')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Plot weighted moving averages for each ETF
for etf in etfs:
    plot_weighted_moving_average(cleaned_data[etf], f'{etf} Weighted Moving Average')

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

def plot_exponential_smoothing(df, title):
    plt.figure(figsize=(14, 8))
    plt.plot(df, label='Adjusted Close')
    
    # Single Exponential Smoothing
    model = SimpleExpSmoothing(df).fit()
    plt.plot(model.fittedvalues, label='Single Exponential Smoothing')
    
    # Double Exponential Smoothing
    model = ExponentialSmoothing(df, trend='add').fit()
    plt.plot(model.fittedvalues, label='Double Exponential Smoothing')
    
    # Triple Exponential Smoothing
    model = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12).fit()
    plt.plot(model.fittedvalues, label='Triple Exponential Smoothing')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Plot exponential smoothing for each ETF
for etf in etfs:
    plot_exponential_smoothing(cleaned_data[etf], f'{etf} Exponential Smoothing')

for etf in etfs:
    print(f'Descriptive Statistics for {etf}:')
    print(cleaned_data[etf].describe())
    print()


#================ARIMA=======

# Split data 70/30 for train/test
split_data = {}
for etf in etfs:
    split_data[etf] = train_test_split(cleaned_data[etf], test_size=0.3, shuffle=False)

#-----------------------MODEL--------------------------
# ARIMA model for each ETF
arima_predictions = {}
arima_forecasts = {}

for etf in etfs:
    train, test = split_data[etf]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    
    arima_predictions[etf] = predictions
    arima_forecasts[etf] = model_fit.forecast(steps=365)  # Forecast 1 year into the future

#=========Plot Predictions vs Actual Values======

for etf in etfs:
    train, test = split_data[etf]
    plt.figure(figsize=(14, 8))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, arima_predictions[etf], label='Predicted')
    plt.title(f'{etf} ARIMA Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#-------------Forecast Future Values---------------

for etf in etfs:
    plt.figure(figsize=(14, 8))
    plt.plot(cleaned_data[etf].index, cleaned_data[etf], label='Adjusted Close')
    plt.plot(arima_forecasts[etf].index, arima_forecasts[etf], label='Forecasted')
    plt.title(f'{etf} Future Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
#---------------ARIMA Metrics-------------------

def prepare_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

lstm_predictions = {}
lstm_forecasts = {}

for etf in etfs:
    train, test = split_data[etf]
    
    # Reshape the data to fit MinMaxScaler requirements
    train = np.array(train).reshape(-1, 1)
    test = np.array(test).reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # Prepare the data
    X_train, y_train = prepare_data(train_scaled, 60)
    X_test, y_test = prepare_data(test_scaled, 60)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
#------------------------------ Build LSTM model-------------------------------
def prepare_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


lstm_predictions = {}
lstm_forecasts = {}

for etf in etfs:
    train, test = split_data[etf]
    
    # Reshape the data to fit MinMaxScaler requirements
    train = np.array(train).reshape(-1, 1)
    test = np.array(test).reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    # Prepare the data
    X_train, y_train = prepare_data(train_scaled, 60)
    X_test, y_test = prepare_data(test_scaled, 60)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'{etf} LSTM Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    lstm_predictions[etf] = predictions
    
    # Forecast future values for 1 year (365 days)
    future_inputs = scaler.transform(np.array(cleaned_data[etf][-60:]).reshape(-1, 1))
    future_inputs = future_inputs.reshape((1, future_inputs.shape[0], 1))
    
    lstm_forecast = []
    for _ in range(365):
        next_pred = model.predict(future_inputs)
        lstm_forecast.append(next_pred[0, 0])
        next_pred_reshaped = np.array(next_pred[0, 0]).reshape(1, 1, 1)  # Reshape to match dimensions
        future_inputs = np.append(future_inputs[:, 1:, :], next_pred_reshaped, axis=1)
    
    lstm_forecasts[etf] = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))


#===================XGBoost============

def create_features(df, label=None):
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week
    df['weekday'] = df.index.weekday
    X = df[['dayofyear', 'month', 'weekofyear', 'weekday']]
    if label:
        y = df[label]
        return X, y
    return X

xgb_predictions = {}
xgb_forecasts = {}

for etf in etfs:
    train, test = split_data[etf]
    X_train, y_train = create_features(train.to_frame(), 'Adj Close')
    X_test, y_test = create_features(test.to_frame(), 'Adj Close')
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    xgb_predictions[etf] = predictions
    future_index = pd.date_range(start=cleaned_data[etf].index[-1], periods=365, freq='B')
    future_features = create_features(pd.DataFrame(index=future_index))
    xgb_forecasts[etf] = model.predict(future_features)

# ------------------Calculate MAE, MSE, RMSE-----------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
metrics_list = []
metrics = pd.DataFrame(columns=['ETF', 'Model', 'MAE', 'MSE', 'RMSE'])

for etf in etfs:
    train, test = split_data[etf]
    arima_mae = mean_absolute_error(test, arima_predictions[etf])
    arima_mse = mean_squared_error(test, arima_predictions[etf])
    arima_rmse = np.sqrt(arima_mse)
    
    lstm_mae = mean_absolute_error(test[60:], lstm_predictions[etf])
    lstm_mse = mean_squared_error(test[60:], lstm_predictions[etf])
    lstm_rmse = np.sqrt(lstm_mse)
    
    xgb_mae = mean_absolute_error(test, xgb_predictions[etf])
    xgb_mse = mean_squared_error(test, xgb_predictions[etf])
    xgb_rmse = np.sqrt(xgb_mse)
    
    metrics_list.append({'ETF': etf, 'Model': 'ARIMA', 'MAE': arima_mae, 'MSE': arima_mse, 'RMSE': arima_rmse})

    # Append values for LSTM model
    metrics_list.append({'ETF': etf, 'Model': 'LSTM', 'MAE': lstm_mae, 'MSE': lstm_mse, 'RMSE': lstm_rmse})

    # Append values for XGBoost model
    metrics_list.append({'ETF': etf, 'Model': 'XGBoost', 'MAE': xgb_mae, 'MSE': xgb_mse, 'RMSE': xgb_rmse})

# Convert the list of dictionaries into a DataFrame
metrics = pd.DataFrame(metrics_list)

print(metrics)
      
# Plot comparison of predictions vs actual values
for etf in etfs:
    train, test = split_data[etf]
    plt.figure(figsize=(14, 8))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, arima_predictions[etf], label='ARIMA')
    plt.plot(test.index[60:], lstm_predictions[etf], label='LSTM')
    plt.plot(test.index, xgb_predictions[etf], label='XGBoost')
    plt.title(f'{etf} Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Display the metrics
metrics.set_index(['ETF', 'Model'], inplace=True)
print(metrics)

# Plotting metrics
metrics.reset_index(inplace=True)
plt.figure(figsize=(14, 8))
sns.barplot(data=metrics, x='ETF', y='RMSE', hue='Model')
plt.title('Model Comparison by RMSE')
plt.ylabel('RMSE')
# Annotate each bar with its value
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(data=metrics, x='ETF', y='MAE', hue='Model')
plt.title('Model Comparison by MAE')
plt.ylabel('MAE')
# Annotate each bar with its value
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(data=metrics, x='ETF', y='MSE', hue='Model')
plt.title('Model Comparison by MSE')
plt.ylabel('MSE')
# Annotate each bar with its value
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
plt.show()


#==================Plot Comparision =========================

for etf in etfs:
    train, test = split_data[etf]
    plt.figure(figsize=(14, 8))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, arima_predictions[etf], label='ARIMA')
    plt.plot(test.index[60:], lstm_predictions[etf], label='LSTM')
    plt.plot(test.index, xgb_predictions[etf], label='XGBoost')
    plt.title(f'{etf} Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#----------------Plot Future Forecasts-----------------

import pandas as pd
import matplotlib.pyplot as plt

# Assume `arima_forecasts`, `xgb_forecasts`, and `lstm_forecasts` are dictionaries with ETF symbols as keys
# and numpy arrays as values.

for etf in etfs:
    # Get the last date from the training data
    last_date = cleaned_data[etf].index[-1]
    
    # Create a date range for the forecast
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)
    
    # Convert the forecast arrays to pandas Series
    arima_forecast_series = pd.Series(arima_forecasts[etf].ravel(), index=future_dates)
    xgb_forecast_series = pd.Series(xgb_forecasts[etf].ravel(), index=future_dates)
    lstm_forecast_series = pd.Series(lstm_forecasts[etf].ravel(), index=future_dates)
    
    # Plot the actual adjusted close prices and the forecasts
    plt.figure(figsize=(14, 8))
    plt.plot(cleaned_data[etf].index, cleaned_data[etf], label='Adjusted Close')
    plt.plot(arima_forecast_series.index, arima_forecast_series, label='ARIMA')
    plt.plot(xgb_forecast_series.index, xgb_forecast_series, label='XGBoost')
    plt.plot(lstm_forecast_series.index, lstm_forecast_series, label='LSTM')
    plt.title(f'{etf} Future Forecast (1 year)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


#==============Plot Cumulative Log Return========================
# Cumulative log return
def plot_cumulative_log_return(df, title):
    log_return = np.log(df / df.shift(1)).cumsum()
    plt.figure(figsize=(14, 8))
    plt.plot(log_return, label='Cumulative Log Return')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.show()

# Plot cumulative log return for each ETF
for etf in etfs:
    plot_cumulative_log_return(cleaned_data[etf], f'{etf} Cumulative Log Return')

#---------------All In One Graph Cumulative Log Return


def plot_cumulative_log_return(df_dict, title):
    plt.figure(figsize=(14, 8))
    
    for etf, df in df_dict.items():
        log_return = np.log(df / df.shift(1)).cumsum()
        plt.plot(log_return, label=etf)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.show()

# Create a dictionary of cleaned data for each ETF
etf_data = {etf: cleaned_data[etf] for etf in etfs}

# Plot cumulative log return for all ETFs in one plot
plot_cumulative_log_return(etf_data, 'Cumulative Log Return for All ETFs')

#===============Portfolio Optimization ============
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Assuming cleaned_data is a dictionary with ETF symbols as keys and their adjusted close prices as values

# Calculate daily returns
returns = pd.DataFrame({etf: cleaned_data[etf].pct_change().dropna() for etf in etfs})

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Annualize the mean returns and covariance matrix
mean_returns_annual = mean_returns * 252
cov_matrix_annual = cov_matrix * 252

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, returns

# Function to minimize (negative Sharpe ratio)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_ret - risk_free_rate) / p_var

# Constraints: Sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds: Weights between 0 and 1
bounds = tuple((0, 1) for _ in range(len(etfs)))

# Initial guess: Equal distribution
initial_weights = [1 / len(etfs)] * len(etfs)

# Risk-free rate (e.g., 0.01 for 1%)
risk_free_rate = 0.01

# Optimization
optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(mean_returns_annual, cov_matrix_annual, risk_free_rate),
                             method='SLSQP', bounds=bounds, constraints=constraints)

# Get the optimal weights
optimal_weights = optimized_results.x

# Calculate expected portfolio performance
portfolio_std_dev, portfolio_return = portfolio_performance(optimal_weights, mean_returns_annual, cov_matrix_annual)
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

# Allocate the ??? Euros based on the optimal weights
initial_investment = 35000
allocation = initial_investment * optimal_weights

# Create a DataFrame for the allocation
allocation_df = pd.DataFrame({'ETF': etfs, 'Optimal Weight': optimal_weights, 'Allocation (Euros)': allocation})

# Display the allocation
print("Portfolio Allocation:\n", allocation_df)

# Round the values in the DataFrame to the nearest integer
allocation_df_rounded = allocation_df.copy()
allocation_df_rounded['Optimal Weight'] = allocation_df_rounded['Optimal Weight'].round(2)
allocation_df_rounded['Allocation (Euros)'] = allocation_df_rounded['Allocation (Euros)'].round(2)
print(allocation_df_rounded)

# Visualization in tabular format using Matplotlib
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=allocation_df_rounded.values, colLabels=allocation_df_rounded.columns, cellLoc='center', loc='center')
table.scale(1, 1.5)
plt.title('Portfolio Allocation Table')
plt.show()

# Visualization as a bar graph
plt.figure(figsize=(10, 6))
bars = plt.bar(allocation_df['ETF'], allocation_df['Allocation (Euros)'])
plt.xlabel('ETF')
plt.ylabel('Allocation (Euros)')
plt.title('Portfolio Allocation')
plt.ylim(0, max(allocation) * 1.1)  # Add some space on top

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.show()
