import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
print(sys.executable)


st.title("AAPL Stock Price Prediction Dashboard")

# User input for ticker and date range
st.header("Select Stock and Date Range")
ticker = st.text_input("Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-07-17"))

# Fetch and process data
if st.button("Load Data"):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Next_Close'] = data['Close'].shift(-1)
    data = data.dropna()

    # Display data
    st.header("Stock Data")
    st.write(f"{ticker} Data Preview:")
    st.dataframe(data.head())

    # Visualize data
    st.header("Stock Price Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], name="SMA_10"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name="SMA_50"))
    fig.update_layout(title=f"{ticker} Closing Prices with Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig)

    # Train model and predict
    features = ['Close', 'Volume', 'SMA_10', 'SMA_50']
    X = data[features]
    y = data['Next_Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display predictions
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=X_test.index)
    predictions = predictions.sort_index()
    st.header("Model Predictions")
    st.write("Predictions Preview:")
    st.dataframe(predictions.head())

    # Visualize predictions
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=predictions.index, y=predictions['Actual'], name="Actual Next_Close"))
    fig_pred.add_trace(go.Scatter(x=predictions.index, y=predictions['Predicted'], name="Predicted Next_Close"))
    fig_pred.update_layout(title=f"Actual vs. Predicted {ticker} Next Close Prices", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_pred)

    # Model metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")