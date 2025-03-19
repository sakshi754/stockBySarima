import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Streamlit App Title
st.title("Stock Price Prediction using SARIMA Model")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")  # Fetch last 1 year of data
    return df

if st.button("Predict"):
    # Fetch stock data
    df = fetch_stock_data(ticker)
    
    if df.empty:
        st.error("Invalid ticker or no data available. Please try again.")
    else:
        # Prepare data for SARIMA
        prices = df['Close'].values

        # Fit SARIMA model (Using SARIMA(1,1,1)(1,1,1,12) as an initial configuration)
        model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))  # (p,d,q) x (P,D,Q,s)
        model_fit = model.fit()

        # Predict the next 30 days
        future_forecast = model_fit.forecast(steps=30)
        
        # Generate future dates
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=30, freq='D')

        # Plot Only Future Predictions
        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, future_forecast, label="Predicted Prices", linestyle='dashed', color="red")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title(f"Stock Price Prediction for {ticker} (Next 30 Days using SARIMA)")
        plt.legend()

        # Display plot in Streamlit
        st.pyplot(plt)
