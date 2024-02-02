import yfinance as yf # Get finals data from yahoo finance
import pandas as pd # is used for data manipulation and analysis
import numpy as np #  is used for numerical operations.
from sklearn.model_selection import train_test_split # Support Vector Machine (SVM) model
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # Create Graphs
import matplotlib.dates as mdates # Dates Format
from ta import add_all_ta_features # Technical Analysis library to financial datasets with Python Pandas
import mplcursors # Cursor Events Hundler

# Function downloads historical stock data using the yfinance library.
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    # Check if necessary columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in stock_data.columns for col in required_columns):
        print("Fetching additional data with 'Open', 'High', 'Low', 'Close', 'Volume' columns.")
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return stock_data.reset_index()

# Function adds moving averages and technical analysis features using the ta library.
def create_features(data):
    # Calculate moving averages (you can customize the window size)
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    # Add technical analysis features using the ta library
    data = add_all_ta_features(data, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
    return data

# Function extracts features and target variable (Close prices) from the data.
def prepare_data(data):
    features = ['Close', 'MA10', 'MA50', 'volume_adi', 'volatility_bbh', 'volatility_bbl', 'volatility_kcc', 'volatility_kch']
    data = create_features(data)
    data = data.dropna()
    X = data[features]
    y = data['Close']
    return X, y

# Function scales features and trains a Support Vector Machine (SVM) model.
def train_svm_model(X_train, y_train):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Train SVM model
    svr = SVR(kernel='linear')
    svr.fit(X_train_scaled, y_train)
    return svr, scaler

#  function makes predictions using the trained SVM model.
def predict_stock_price(svr, scaler, X_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = svr.predict(X_test_scaled)
    return y_pred

# Function plots the history stock prices (Display Graph). 
def plot_stock_history(stock_data, test_start_index):
    one_month_before_start = max(0, test_start_index - 30)
    plt.plot(
        stock_data['Date'][one_month_before_start:test_start_index],
        stock_data['Close'][one_month_before_start:test_start_index],
        label='One Month Before',
        color='green')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function plots the actual stock prices (Display Graph).
def plot_actual(actual):
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function plots the predicted stock prices (Display Graph). 
def plot_predicted(predicted):
    plt.plot(predicted, label='Predicted Prices', color='red')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
format_point_over_str = lambda sel: pd.to_datetime(sel.target[0], unit='D')
    
def plot_combined_results(stock_data, actual, predicted, test_start_index):
    one_month_before_start = max(0, test_start_index - 30)
    test_end_index = test_start_index + 30
    
    # Calculate and plot differences betoween actual and predicted
    differences = actual - predicted
    
    # Convert numeric date values to Pandas Timestamps
    date_one_month_before = stock_data['Date'][one_month_before_start:test_end_index].apply(pd.Timestamp)
    date_actual = stock_data['Date'][test_start_index:test_end_index].apply(pd.Timestamp)
    date_predicted = stock_data['Date'][test_start_index:test_end_index].apply(pd.Timestamp)    
    date_difrences = stock_data['Date'][test_start_index:test_start_index+len(differences)].apply(pd.Timestamp)

    plt.figure(figsize=(14, 7))  # Adjust the figsize based on your screen resolution

    # pd.Timestamp(sel.target[0]).strftime('%Y-%m-%d')

    # Plot stock history for one month before the test period
    plt.subplot(4, 1, 1)
    plt.plot(date_one_month_before,
             stock_data['Close'][one_month_before_start:test_end_index],
             label='One Month Before',
             color='green')
    plt.xlabel('Date')
    plt.ylabel('One Month Before')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    mplcursors.cursor(hover=True).connect(
        "add", 
        format_point_over_str)
    plt.legend()
    

    # Plot actual prices
    plt.subplot(4, 1, 2)
    plt.plot(
        date_actual,
        actual[:len(stock_data['Date'][test_start_index:test_end_index])],
        label='Actual Prices',
        color='blue')
    plt.xlabel('Date')
    plt.ylabel('Actual Prices')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    mplcursors.cursor(hover=True).connect(
        "add", 
        format_point_over_str)
    plt.legend()
    

    # Plot predicted prices
    plt.subplot(4, 1, 3)
    plt.plot(
        date_predicted,
        predicted[:len(stock_data['Date'][test_start_index:test_end_index])],
        label='Predicted Prices',
        color='red')
    plt.xlabel('Date')
    plt.ylabel('Predicted Prices')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    mplcursors.cursor(hover=True).connect(
        "add",
        format_point_over_str)
    plt.legend()
    

    # Plot differences betoween actual and predicted
    plt.subplot(4, 1, 4)
    plt.plot(
        date_difrences,
        differences,
        label='Differences (Actual - Predicted)',
        color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')
    plt.title('Differences between Actual and Predicted Prices')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    mplcursors.cursor(hover=True).connect(
        "add",
        format_point_over_str)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stock_symbol = 'AAPL'  # Replace with the desired stock symbol
    start_date = '2023-01-01'  # Replace with your desired start date
    end_date = '2024-01-01'  # Replace with your desired end date

    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    X, y = prepare_data(stock_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    svr_model, scaler_model = train_svm_model(X_train, y_train)

    # Make predictions
    y_pred = predict_stock_price(svr_model, scaler_model, X_test)

    # Plot combined results
    test_start_index = stock_data.index.get_loc(X_test.index[0])
    plot_combined_results(stock_data, y_test.values, y_pred, test_start_index)
