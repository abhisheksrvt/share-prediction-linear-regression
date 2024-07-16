# Disclaimer: This project is for educational and research purposes only. 
# It is strictly not intended for use in making financial decisions, such as buying or selling stocks. 
# The predictions made by this model are not financial advice and should not be treated as such.

import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1), data.index

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
    
    # Save scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    np.save('X.npy', X)
    np.save('y.npy', y)
    
    start_date = pd.Timestamp(dates[0])
    end_date = pd.Timestamp(dates[-1])
    
    return X, y, scaler, start_date, end_date

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

    # Save the model
    with open('stock_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Model training complete. Saved the model to stock_model.pkl.')
    
    return model

def predict(model, X, future_days=30):
    # Predict using the last sequence in X
    last_sequence = X[-1].reshape(1, -1)
    future_predictions = []
    
    for _ in range(future_days):
        prediction = model.predict(last_sequence)[0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1][-1] = prediction
    
    return future_predictions

def plot_results(data, y_pred, future_pred, dates, ticker):
    plt.figure(figsize=(12, 8))
    
    # Plot true prices
    plt.subplot(2, 1, 1)
    plt.plot(dates, data, label='True Prices', linestyle='-', linewidth=1.5, color='blue')
    plt.legend()
    plt.title(ticker)
    plt.grid(True)
    
    # Plot predicted prices
    predicted_start = len(data) - len(y_pred)
    predicted_end = len(data)
    plt.plot(dates[predicted_start:predicted_end], y_pred, label='Predicted Prices', linestyle='-', linewidth=1.0, color='orange')
    plt.legend()
    plt.grid(True)
    
    # Plot future predicted prices
    plt.subplot(2, 1, 2)
    future_dates = pd.date_range(start=dates[-1], periods=len(future_pred)+1, freq='B')[1:]
    plt.plot(future_dates, future_pred, label='Future Predictions', linestyle='--', linewidth=1.0, color='green')
    plt.legend(loc='upper left')  # Set legend to upper left
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_future_predictions(future_predictions, start_date, end_date):
    last_date = pd.Timestamp(end_date) 
    print("\nPredicted Prices for the next 30 days:")
    for i, prediction in enumerate(future_predictions, start=1):
        next_date = last_date + pd.DateOffset(days=i)
        print(f"{next_date.date()}: {prediction:.2f}")

def main():
    ticker = 'TCS.NS'  # Change to the desired NSE stock symbol
    start_date = '2010-01-01'  # Default start date (can be overridden)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Automatically set end date to today
    look_back = 60
    
    # Download data
    data, dates = download_data(ticker, start_date, end_date)
    
    # Preprocess data and retrieve start_date and end_date from data
    X, y, scaler, start_date, end_date = preprocess_data(data, dates, look_back)
    
    # Check if data is available for training
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available for training.")
    
    # Train model
    model = train_model(X, y)
    
    # Predict future prices
    future_predictions = predict(model, X, future_days=30)
    
    # Inverse transform all y values for printing
    y_pred = scaler.inverse_transform(model.predict(X).reshape(-1, 1)).flatten()
    future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    # Print future predictions along with dates
    print_future_predictions(future_pred, start_date, end_date)
    # Plotting results with dates on x-axis
    plot_results(data, y_pred, future_pred, dates, ticker)

if __name__ == "__main__":
    main()

# Training Results -
# MAE: 0.0049888276624443185
# MSE: 5.880408938968448e-05
# RMSE: 0.007668382449362087
