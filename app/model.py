import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import data as data_managment
import joblib
from tensorflow.keras.models import save_model, load_model
import matplotlib.pyplot as plt

sequence_length = 10

def create_blank_models(X_element_shape, sequence_length=sequence_length):
    lstm_model = Sequential([
        LSTM(64, input_shape=(sequence_length, X_element_shape), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    return lstm_model, xgb_model

def train_model(data):
    features = data[:, :-1]
    target = data[:, -1]
    shape = len(features[0])

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences for LSTM
    def create_sequences(data, target):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:i+sequence_length]
            y = target[i+sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    sequence_length = 10
    X, y = create_sequences(features_scaled, target)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model, xgb_model = create_blank_models(shape)

    history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

    # Extract LSTM features
    lstm_features_train = lstm_model.predict(X_train)
    lstm_features_test = lstm_model.predict(X_test)

    # Combine LSTM features with other inputs
    additional_features_train = features[sequence_length:len(X_train) + sequence_length]
    additional_features_test = features[len(X_train) + sequence_length:]

    X_train_combined = np.hstack((lstm_features_train, additional_features_train))
    X_test_combined = np.hstack((lstm_features_test, additional_features_test))

    # XGBoost Model
    xgb_model.fit(X_train_combined, y_train)

    y_pred = xgb_model.predict(X_test_combined)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')

    # Generowanie wykresu z wynikami trenowania
    plot_training_results(y_test, y_pred, history)

    return lstm_model, xgb_model, scaler

def plot_training_results(y_test, y_pred, history):
    plt.figure(figsize=(14, 7))

    # wartości straty trenowania i walidacji
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Strata modelu')
    plt.ylabel('Strata')
    plt.xlabel('Epoka')
    plt.legend(['Trenowanie', 'Walidacja'], loc='upper left')

    # rzeczywiste vs przewidywane wartości
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Rzeczywiste')
    plt.plot(y_pred, label='Przewidywane')
    plt.title('Rzeczywiste vs Przewidywane')
    plt.ylabel('Wartość')
    plt.xlabel('Próbka')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def predict_target(single_X, lstm_model, xgb_model, scaler):
    single_X = np.array(single_X).reshape(1, -1)
    single_X_scaled = scaler.transform(single_X)
    lstm_input = single_X_scaled.reshape(1, 1, -1)
    lstm_features = lstm_model.predict(lstm_input)
    combined_features = np.hstack((lstm_features, single_X_scaled))
    target_pred = xgb_model.predict(combined_features)
    return target_pred

def save_models(lstm_model, xgb_model, scaler, lstm_path='lstm_model.h5', xgb_path='xgb_model.joblib', scaler_path='scaler.joblib'):
    save_model(lstm_model, lstm_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(scaler, scaler_path)
    print(f'Models and scaler saved to {lstm_path}, {xgb_path}, and {scaler_path}')

def load_models(lstm_path='lstm_model.h5', xgb_path='xgb_model.joblib', scaler_path='scaler.joblib'):
    lstm_model = load_model(lstm_path)
    xgb_model = joblib.load(xgb_path)
    scaler = joblib.load(scaler_path)
    print(f'Models and scaler loaded from {lstm_path}, {xgb_path}, and {scaler_path}')
    return lstm_model, xgb_model, scaler
