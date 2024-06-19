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
from keras.optimizers import Adam
from datetime import datetime, timedelta
import data

sequence_length = 10

def create_sequences(data, target):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:i + sequence_length]
        y = target[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_blank_models(X_element_shape, sequence_length=sequence_length):
    lstm_model = Sequential([
        LSTM(64, input_shape=(sequence_length, X_element_shape), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
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

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    sequence_length = 10
    X, y = create_sequences(features_scaled, target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model, xgb_model = create_blank_models(shape)
    history = lstm_model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.1)

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

    return lstm_model, xgb_model, scaler, mse, mae, mape

def retrain_models(data, lstm_model, xgb_model, scaler):
    features = data[:, :-1]
    target = data[:, -1]

    # Normalize features using the existing scaler
    features_scaled = scaler.transform(features)

    X, y = create_sequences(features_scaled, target)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lstm_model.compile(optimizer=Adam(), loss='mean_squared_error')
    history = lstm_model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.1)

    # Extract LSTM features
    lstm_features_train = lstm_model.predict(X_train)
    lstm_features_test = lstm_model.predict(X_test)

    # Combine LSTM features with other inputs
    additional_features_train = features[sequence_length:len(X_train) + sequence_length]
    additional_features_test = features[len(X_train) + sequence_length:]

    X_train_combined = np.hstack((lstm_features_train, additional_features_train))
    X_test_combined = np.hstack((lstm_features_test, additional_features_test))

    # Continue training XGBoost Model
    xgb_model.fit(X_train_combined, y_train)

    y_pred = xgb_model.predict(X_test_combined)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Absolute Percentage Error: {mape}')

    plot_training_results(y_test, y_pred, history)
    return lstm_model, xgb_model, scaler, mse, mae, mape

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
    plt.savefig('static/training_results.png')
    plt.show()

def predict_target_from_single_date(X_row, lstm_model, xgb_model, scaler, sequence_length=sequence_length):
    # Normalize the single row
    X_row_scaled = scaler.transform([X_row])
    # Create a sequence from the single row by repeating it
    X_sequence = np.array([X_row_scaled] * sequence_length)
    X_sequence = X_sequence.reshape((1, sequence_length, -1))  # Reshape to (1, sequence_length, num_features)
    # Predict LSTM features
    lstm_features = lstm_model.predict(X_sequence)[0]
    # Combine LSTM features with the original input row (normalized)
    combined_features = np.hstack((lstm_features, X_row_scaled[0]))
    # Predict using XGBoost
    y_pred = xgb_model.predict(np.array([combined_features]))
    return y_pred[0]

def generate_date_range(date_start, date_end):
    start_date = datetime.strptime(date_start, "%d-%m-%Y")
    end_date = datetime.strptime(date_end, "%d-%m-%Y")
    date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    return date_list

def predict_target_from_date_range(date_start, date_end, lstm_model, xgb_model, scaler, mineral, sequence_length=10):
    try:
        date_range = generate_date_range(date_start, date_end)
        data_packets = []

        # Generate data for each date in the range
        for current_date in date_range:
            data_packet = data.generate_singe_data_packet(current_date, mineral, with_target=False)
            data_packets.append(data_packet)

        # Convert to numpy array and scale
        data_packets = np.array(data_packets)
        scaled_data = scaler.transform(data_packets)

        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - sequence_length + 1):
            sequences.append(scaled_data[i:i + sequence_length])

        sequences = np.array(sequences)

        # Predict with LSTM model
        lstm_predictions = lstm_model.predict(sequences)

        # Check for NaN or infinite values in LSTM predictions
        if np.any(np.isnan(lstm_predictions)) or np.any(np.isinf(lstm_predictions)):
            raise ValueError("LSTM predictions contain NaN or infinite values.")

        # Flatten sequences for XGBoost model
        xgb_input = sequences.reshape((sequences.shape[0], -1))

        # Ensure the input shape matches XGBoost model's expected input shape
        expected_shape = xgb_model.get_booster().num_features
        if xgb_input.shape[1] != expected_shape:
            raise ValueError(f"Expected input shape: ({xgb_input.shape[0]}, {expected_shape}), but got: {xgb_input.shape}")

        # Predict with XGBoost model
        xgb_predictions = xgb_model.predict(xgb_input)

        # Check for NaN or infinite values in XGBoost predictions
        if np.any(np.isnan(xgb_predictions)) or np.any(np.isinf(xgb_predictions)):
            raise ValueError("XGBoost predictions contain NaN or infinite values.")

        # Combine predictions (if required, this is a simple average example)
        combined_predictions = (lstm_predictions.flatten() + xgb_predictions) / 2

        return combined_predictions

    except ValueError as e:
        print(f"ValueError in prediction: {e}")
        return None
    except AttributeError as e:
        print(f"AttributeError in prediction: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in prediction: {e}")
        return None


def save_models(lstm_model, xgb_model, scaler, lstm_path='lstm_model.h5', xgb_path='xgb_model.joblib', scaler_path='scaler.joblib'):
    # Save the LSTM model
    save_model(lstm_model, lstm_path)

    # Save the XGBoost model
    joblib.dump(xgb_model, xgb_path)

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    print(f'Models and scaler saved to {lstm_path}, {xgb_path}, and {scaler_path}')

# Example usage:
# save_models(lstm_model, xgb_model, scaler)


def load_models(lstm_path='lstm_model.h5', xgb_path='xgb_model.joblib', scaler_path='scaler.joblib'):
    # Load the LSTM model
    lstm_model = load_model(lstm_path)

    # Load the XGBoost model
    xgb_model = joblib.load(xgb_path)

    # Load the scaler
    scaler = joblib.load(scaler_path)

    print(f'Models and scaler loaded from {lstm_path}, {xgb_path}, and {scaler_path}')

    return lstm_model, xgb_model, scaler

# Example usage:
# lstm_model, xgb_model, scaler = load_models()
