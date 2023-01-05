import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def price_predict():
    # Load Data
    company = 'DIS'

    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 1, 1)

    data = web.DataReader(company, 'yahoo', start, end)

    # Preparing Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 30

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=7))  # Predicts closing price of the next day

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=36)

    ''' Test Model Accuracy on Existing Data '''

    # Load Test Data
    test_start = dt.datetime(2021, 1, 1)
    test_end = dt.datetime(2022, 12, 1)

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Real Data
    real_data = []

    for x in range(prediction_days, len(model_inputs)):
        real_data.append(model_inputs[x - prediction_days:x, 0])

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    predicted_prices = model.predict(real_data)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict Next Day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    print(f"Next Day Prediction: {prediction} ")

    # Plot the Test Predictions
    plt.plot(actual_prices, color="black", label=f'Actual {company} Price')
    plt.plot(predicted_prices, color='Red', label=f'Predicted {company} Price')
    plt.title(f"{company} Share Price ")
    plt.xlabel('Time')
    plt.ylabel(f"{company} Share Price ")
    plt.legend()
    plt.show()

    # price_start = dt.datetime(2022, 12, 2)
    # price_end = dt.datetime(2022, 12, 2)
    # data_price = web.DataReader(company, 'yahoo', price_start, price_end)
    # price = data_price['Adj Close'].values
    # accuracy = (abs((prediction - price))/ price) * 100
    # print(f'This prediction is: {accuracy} % accurate')


if __name__ == "__main__":
    price_predict()
