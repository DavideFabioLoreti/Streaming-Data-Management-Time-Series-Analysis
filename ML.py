import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- SEED ---
np.random.seed(13)
tf.random.set_seed(13)

# --- CONFIG ---
DATASET_PATH = r"student_dataset.csv"
HOLIDAY_COUNTRY = 'AU'
N_STEPS = 72
EPOCHS = 15
BATCH_SIZE = 64
TEST_RATIO = 0.1
FORECAST_HORIZON = 1439

# --- LOAD DATA ---
df = pd.read_csv(DATASET_PATH)
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.set_index('time').sort_index()
df = df.dropna(subset=['value']).copy()

# --- FEATURE ENGINEERING ---
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
au_holidays = holidays.CountryHoliday(HOLIDAY_COUNTRY)
df['is_holiday'] = df.index.to_series().apply(lambda d: 1 if d.date() in au_holidays else 0)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/23)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/23)

df = df.drop(columns=['hour','dayofweek'])
input_features = [c for c in df.columns if c != 'value'] + ['value']

data = df[input_features].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
TARGET_COL = scaled.shape[1]-1
n_features = scaled.shape[1]

# --- CREATE SEQUENCES ---
def split_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, :])
        y.append(data[i+n_steps, TARGET_COL])
    return np.array(X), np.array(y)

X, y = split_sequences(scaled, N_STEPS)

# --- TRAIN / TEST SPLIT ---
idx = int(len(X)*(1-TEST_RATIO))
X_train, X_test = X[:idx], X[idx:]
y_train, y_test = y[:idx], y[idx:]

# --- LSTM MODEL ---
lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(N_STEPS, n_features)),
    Dropout(0.2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# --- PREDICTIONS ON TEST SET ---
pred_lstm_scaled = lstm.predict(X_test).flatten()
aux_test = X_test[:, -1, :-1]

def inverse_target(pred_scaled, aux_features):
    return scaler.inverse_transform(np.hstack([aux_features, pred_scaled.reshape(-1,1)]))[:, TARGET_COL]

pred_lstm = inverse_target(pred_lstm_scaled, aux_test)
y_test_true = inverse_target(y_test, aux_test)

# --- KNN MODEL ---
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_flat, y_train)
pred_knn_scaled = knn.predict(X_test_flat)
pred_knn = inverse_target(pred_knn_scaled, aux_test)

# --- RANDOM FOREST MODEL ---
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=13)
rf.fit(X_train_flat, y_train)
pred_rf_scaled = rf.predict(X_test_flat)
pred_rf = inverse_target(pred_rf_scaled, aux_test)

# --- XGBOOST MODEL ---
xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8)
xgb.fit(X_train_flat, y_train)
pred_xgb_scaled = xgb.predict(X_test_flat)
pred_xgb = inverse_target(pred_xgb_scaled, aux_test)

# --- METRICS ---
MAE_LSTM = mean_absolute_error(y_test_true, pred_lstm)
MAE_KNN = mean_absolute_error(y_test_true, pred_knn)
MAE_RF = mean_absolute_error(y_test_true, pred_rf)
MAE_XGB = mean_absolute_error(y_test_true, pred_xgb)

print("\n===== MODEL METRICS =====")
print(f"LSTM:        MAE={MAE_LSTM:.3f}")
print(f"KNN:         MAE={MAE_KNN:.3f}")
print(f"RandomForest MAE={MAE_RF:.3f}")
print(f"XGBoost:     MAE={MAE_XGB:.3f}")

# --- PLOT COMPARISON ON TEST SET ---
plt.figure(figsize=(14,6))
history = y_test_true[-200:]
plt.plot(range(len(history)), history, label='Actual', linewidth=2, color='black')
plt.plot(range(len(history)), pred_lstm[-200:], label='LSTM', alpha=0.7)
plt.plot(range(len(history)), pred_knn[-200:], label='KNN', alpha=0.7)
plt.plot(range(len(history)), pred_rf[-200:], label='RandomForest', alpha=0.7)
plt.plot(range(len(history)), pred_xgb[-200:], label='XGBoost', alpha=0.7)
plt.title('Model Forecast Comparison (Test Set, Last 200 Steps)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# --- RECURSIVE FORECAST ---
def recursive_forecast(model, last_seq, start_time, n_steps, horizon, model_type='sklearn'):
    """
    Recursive forecasting con aggiornamento dinamico delle feature temporali.
    """
    preds = []
    seq = last_seq.copy()
    current_time = start_time

    for step in range(horizon):
        if model_type == 'lstm':
            inp = seq.reshape(1, n_steps, n_features)
            pred = model.predict(inp, verbose=0)[0, 0]
        else:
            X_last = seq.flatten().reshape(1, -1)
            pred = model.predict(X_last)[0]

        preds.append(pred)

        current_time += pd.Timedelta(hours=1)
        hour = current_time.hour
        hour_sin = np.sin(2 * np.pi * hour / 23)
        hour_cos = np.cos(2 * np.pi * hour / 23)
        is_weekend = 1 if current_time.dayofweek >= 5 else 0
        is_holiday = 1 if current_time.date() in au_holidays else 0

        new_row_unscaled = np.array([[is_weekend, is_holiday, hour_sin, hour_cos, 0]])
        new_row_scaled = scaler.transform(new_row_unscaled)[0]
        new_step = new_row_scaled.copy()
        new_step[TARGET_COL] = pred

        seq = np.vstack([seq[1:], new_step])

        if step % 200 == 0:
            print(f"  {model_type.upper()}: step {step}/{horizon}")

    return np.array(preds)

# --- GENERATE FUTURE FORECASTS ---
print("\n===== GENERATING FUTURE FORECASTS =====")
last_seq_full = scaled[-N_STEPS:].copy()
start_time = df.index[-1]
horizon = FORECAST_HORIZON

pred_lstm_future = recursive_forecast(lstm, last_seq_full, start_time, N_STEPS, horizon, model_type='lstm')
pred_knn_future = recursive_forecast(knn, last_seq_full, start_time, N_STEPS, horizon, model_type='sklearn')
pred_rf_future = recursive_forecast(rf, last_seq_full, start_time, N_STEPS, horizon, model_type='sklearn')
pred_xgb_future = recursive_forecast(xgb, last_seq_full, start_time, N_STEPS, horizon, model_type='sklearn')

# --- INVERSE TRANSFORM ---
future_times = pd.date_range(start=start_time, periods=horizon+1, freq='H')[1:]
aux_future_list = []
for t in future_times:
    hour = t.hour
    hour_sin = np.sin(2 * np.pi * hour / 23)
    hour_cos = np.cos(2 * np.pi * hour / 23)
    is_weekend = 1 if t.dayofweek >= 5 else 0
    is_holiday = 1 if t.date() in au_holidays else 0
    aux_future_list.append([is_weekend, is_holiday, hour_sin, hour_cos])

aux_future = np.array(aux_future_list)

pred_lstm_future = inverse_target(pred_lstm_future, aux_future)
pred_knn_future = inverse_target(pred_knn_future, aux_future)
pred_rf_future = inverse_target(pred_rf_future, aux_future)
pred_xgb_future = inverse_target(pred_xgb_future, aux_future)

pred_lstm_future = np.maximum(pred_lstm_future, 0)
pred_knn_future = np.maximum(pred_knn_future, 0)
pred_rf_future = np.maximum(pred_rf_future, 0)
pred_xgb_future = np.maximum(pred_xgb_future, 0)

print("\nForecast generation completed!")

# --- PLOT FUTURE FORECASTS ---
plt.figure(figsize=(16,7))
plt.plot(df.index, df['value'], label='Actual Series', color='black', linewidth=1.5, alpha=0.7)
future_index = pd.date_range(start=df.index[-1], periods=horizon+1, freq='H')[1:]
plt.plot(future_index, pred_lstm_future, label='LSTM Forecast', linestyle='--', linewidth=1.5)
plt.plot(future_index, pred_knn_future, label='KNN Forecast', linestyle='--', linewidth=1.5)
plt.plot(future_index, pred_rf_future, label='RandomForest Forecast', linestyle='--', linewidth=1.5)
plt.plot(future_index, pred_xgb_future, label='XGBoost Forecast', linestyle='--', linewidth=1.5)
plt.axvline(x=df.index[-1], color='red', linestyle=':', linewidth=2, alpha=0.5, label='Forecast Start')
plt.title(f"Future Forecasts with Dynamic Temporal Features ({horizon} Steps Ahead)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# --- ZOOM 168 STEP ---
plt.figure(figsize=(14,6))
zoom_horizon = min(168, horizon)
plt.plot(df.index[-168:], df['value'].iloc[-168:], label='Actual (Last Week)', color='black', linewidth=2)
plt.plot(future_index[:zoom_horizon], pred_lstm_future[:zoom_horizon], label='LSTM', linestyle='--', linewidth=1.5)
plt.plot(future_index[:zoom_horizon], pred_knn_future[:zoom_horizon], label='KNN', linestyle='--', linewidth=1.5)
plt.plot(future_index[:zoom_horizon], pred_rf_future[:zoom_horizon], label='RandomForest', linestyle='--', linewidth=1.5)
plt.plot(future_index[:zoom_horizon], pred_xgb_future[:zoom_horizon], label='XGBoost', linestyle='--', linewidth=1.5)
plt.axvline(x=df.index[-1], color='red', linestyle=':', linewidth=2, alpha=0.5)
plt.title(f"Forecast Detail - First Week")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# --- FORECAST STATISTICS ---
print("\n===== FORECAST STATISTICS =====")
print(f"LSTM:        Mean={pred_lstm_future.mean():.2f}, Std={pred_lstm_future.std():.2f}, Min={pred_lstm_future.min():.2f}, Max={pred_lstm_future.max():.2f}")
print(f"KNN:         Mean={pred_knn_future.mean():.2f}, Std={pred_knn_future.std():.2f}, Min={pred_knn_future.min():.2f}, Max={pred_knn_future.max():.2f}")
print(f"RandomForest Mean={pred_rf_future.mean():.2f}, Std={pred_rf_future.std():.2f}, Min={pred_rf_future.min():.2f}, Max={pred_rf_future.max():.2f}")
print(f"XGBoost:     Mean={pred_xgb_future.mean():.2f}, Std={pred_xgb_future.std():.2f}, Min={pred_xgb_future.min():.2f}, Max={pred_xgb_future.max():.2f}")
print(f"\nActual data: Mean={df['value'].mean():.2f}, Std={df['value'].std():.2f}, Min={df['value'].min():.2f}, Max={df['value'].max():.2f}")

# --- SAVE RESULTS ---
results_df = pd.DataFrame({
    'timestamp': future_index,
    'LSTM': pred_lstm_future,
    'KNN': pred_knn_future,
    'RandomForest': pred_rf_future,
    'XGBoost': pred_xgb_future
})
# results_df.to_csv('future_forecasts.csv', index=False)
# print("\nResults saved to 'future_forecasts.csv'")

input("\nPress Enter to exit and close all plots...")
