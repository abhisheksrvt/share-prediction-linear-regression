# Stock Price Forecasting using Linear Regression

This project uses a linear regression model to forecast stock prices. It leverages historical stock price data downloaded from Yahoo Finance 
to train and test the model.

# Disclaimer
This project is for educational and research purposes only. It is strictly not intended for use in making financial decisions, 
such as buying or selling stocks. The predictions made by this model are not financial advice and should not be treated as such.

## Project Overview

The project consists of the following key steps:

1. **Download Historical Stock Price Data**: 
   - Using the Yahoo Finance API, the project downloads historical closing prices for a specified stock ticker within a given date range.

2. **Data Preprocessing**:
   - The downloaded data is normalized using MinMaxScaler to scale the prices between 0 and 1.
   - The data is then formatted into sequences suitable for training a machine learning model. Each sequence includes a
   - specified number of past prices (`look_back` period) to predict the next price.

3. **Model Training**:
   - A linear regression model is trained using the preprocessed data.
   - The model's performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

4. **Future Price Prediction**:
   - The trained model is used to predict future stock prices for a specified number of days.
   - The predicted prices are transformed back to the original scale using the previously saved MinMaxScaler.

5. **Results Visualization**:
   - The true historical prices, predicted prices during the training period, and future predicted prices are plotted for visualization.
   - The predicted future prices are printed along with their respective dates.
  
![tcs](https://github.com/user-attachments/assets/c19e9ac6-fab6-4426-90ef-97accc992272)

## Features

- Downloads historical stock price data
- Preprocesses data using MinMaxScaler
- Trains a linear regression model
- Makes future stock price predictions
- Visualizes the true, predicted, and future stock prices

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/abhisheksrvt/share-prediction-linear-regression.git
    cd share-prediction-linear-regression
    ```

2. Install the required Python packages:

    ```bash
    pip install numpy pandas yfinance scikit-learn matplotlib pickle-mixin
    ```

## Usage

1. Run the script:

    ```bash
    python share_price_forecasting.py
    ```

2. The script will:
    - Download historical stock price data for the specified ticker.
    - Preprocess the data.
    - Train a linear regression model.
    - Predict future stock prices for the next 30 days.
    - Plot the true, predicted, and future stock prices.
    - Print the future stock price predictions.

## Code Explanation

### Import Libraries

```python
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt
import pandas as pd
```python

# Download Historical Stock Price Data
```python
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1), data.index

Downloads historical stock price data from Yahoo Finance.
Returns closing prices and their corresponding dates.
```python
# Preprocess Data
```python
def preprocess_data(data, dates, look_back=60):
    if len(data) == 0:
        raise ValueError("No data downloaded. Please check your data source.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
        
    X, y = np.array(X), np.array(y)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    np.save('X.npy', X)
    np.save('y.npy', y)
    
    start_date = pd.Timestamp(dates[0])
    end_date = pd.Timestamp(dates[-1])
    
    return X, y, scaler, start_date, end_date
```python
Normalizes the data using MinMaxScaler.
Creates sequences of past prices to predict the next price.
Saves the scaler and preprocessed data for future use.

# Train the Model
```python
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

    with open('stock_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Model training complete. Saved the model to stock_model.pkl.')
    
    return model
```python
Trains a linear regression model using the preprocessed data.
Evaluates the model's performance using MAE, MSE, and RMSE.
Saves the trained model.

# Predict Future Prices
```python
def predict(model, X, future_days=30):
    last_sequence = X[-1].reshape(1, -1)
    future_predictions = []
    
    for _ in range(future_days):
        prediction = model.predict(last_sequence)[0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1][-1] = prediction
    
    return future_predictions
```python
Uses the trained model to predict future stock prices.
Generates predictions for the specified number of future days.

# Visualize Results
```python
def plot_results(data, y_pred, future_pred, dates, ticker):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, data, label='True Prices', linestyle='-', linewidth=1.5, color='blue')
    plt.legend()
    plt.title(ticker)
    plt.grid(True)
    
    predicted_start = len(data) - len(y_pred)
    predicted_end = len(data)
    plt.plot(dates[predicted_start:predicted_end], y_pred, label='Predicted Prices', linestyle='-', linewidth=1.0, color='orange')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    future_dates = pd.date_range(start=dates[-1], periods=len(future_pred)+1, freq='B')[1:]
    plt.plot(future_dates, future_pred, label='Future Predictions', linestyle='--', linewidth=1.0, color='green')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```python
Plots the true historical prices, predicted prices during the training period, and future predicted prices.

# Print Future Predictions
```python
def print_future_predictions(future_predictions, start_date, end_date):
    last_date = pd.Timestamp(end_date) 
    print("\nPredicted Prices for the next 30 days:")
    for i, prediction in enumerate(future_predictions, start=1):
        next_date = last_date + pd.DateOffset(days=i)
        print(f"{next_date.date()}: {prediction:.2f}")
```python
Prints the future predicted prices along with their respective dates.

# Main Function
```python
def main():
    ticker = 'RELIANCE.NS'
    start_date = '2010-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    look_back = 1
    
    data, dates = download_data(ticker, start_date, end_date)
    
    X, y, scaler, start_date, end_date = preprocess_data(data, dates, look_back)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available for training.")
    
    model = train_model(X, y)
    
    future_predictions = predict(model, X, future_days=30)
    
    y_pred = scaler.inverse_transform(model.predict(X).reshape(-1, 1)).flatten()
    future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    print_future_predictions(future_pred, start_date, end_date)
    plot_results(data, y_pred, future_pred, dates, ticker)

if __name__ == "__main__":
    main()
```python

# The main function orchestrates the entire workflow, from downloading data to making predictions and visualizing results.

# License
This project is licensed under the MIT License - see the LICENSE file for details.


