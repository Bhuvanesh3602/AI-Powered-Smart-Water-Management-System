import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os

try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using Random Forest only")


class WaterUsagePredictor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lookback = 7
        self.rf_model_file = "rf_predictor.pkl"
        self.lstm_model_file = "lstm_predictor.h5"
        self.scaler_file = "predictor_scalers.pkl"
        
    def prepare_time_series_data(self, df, flow_column, lookback=7):
        daily_usage = df.copy()
        
        if 'timestamp' in df.columns:
            daily_usage['date'] = df['timestamp'].dt.date
            daily_usage = daily_usage.groupby('date')[flow_column].sum().reset_index()
            daily_usage.columns = ['date', 'usage']
        else:
            daily_usage = df[[flow_column]].copy()
            daily_usage.columns = ['usage']
        
        values = daily_usage['usage'].values.reshape(-1, 1)
        
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i, 0])
            y.append(values[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, daily_usage
    
    def train_random_forest(self, df, flow_column):
        X, y, daily_usage = self.prepare_time_series_data(df, flow_column, self.lookback)
        
        if len(X) == 0:
            print("Not enough data for training")
            return None
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        self.save_model()
        
        return metrics
    
    def train_lstm(self, df, flow_column, epochs=50, batch_size=32):
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Using Random Forest instead.")
            return self.train_random_forest(df, flow_column)
        
        X, y, daily_usage = self.prepare_time_series_data(df, flow_column, self.lookback)
        
        if len(X) == 0:
            print("Not enough data for training")
            return None
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        train_pred_scaled = self.model.predict(X_train, verbose=0)
        test_pred_scaled = self.model.predict(X_test, verbose=0)
        
        train_pred = self.scaler_y.inverse_transform(train_pred_scaled).flatten()
        test_pred = self.scaler_y.inverse_transform(test_pred_scaled).flatten()
        y_train_orig = self.scaler_y.inverse_transform(y_train).flatten()
        y_test_orig = self.scaler_y.inverse_transform(y_test).flatten()
        
        metrics = {
            'train_mse': mean_squared_error(y_train_orig, train_pred),
            'test_mse': mean_squared_error(y_test_orig, test_pred),
            'train_mae': mean_absolute_error(y_train_orig, train_pred),
            'test_mae': mean_absolute_error(y_test_orig, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train_orig, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, test_pred)),
            'history': history.history
        }
        
        self.save_model()
        
        return metrics
    
    def predict_next_days(self, df, flow_column, days=7):
        if self.model is None:
            print("Model not trained. Training now...")
            if self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                self.train_lstm(df, flow_column)
            else:
                self.train_random_forest(df, flow_column)
        
        X, y, daily_usage = self.prepare_time_series_data(df, flow_column, self.lookback)
        
        if len(X) == 0:
            return None
        
        last_sequence = X[-1]
        predictions = []
        
        for _ in range(days):
            if self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                seq_scaled = self.scaler_X.transform(last_sequence.reshape(1, -1))
                seq_scaled = seq_scaled.reshape((1, self.lookback, 1))
                pred_scaled = self.model.predict(seq_scaled, verbose=0)
                pred = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
            else:
                pred = self.model.predict(last_sequence.reshape(1, -1))[0]
            
            predictions.append(pred)
            
            last_sequence = np.append(last_sequence[1:], pred)
        
        return predictions
    
    def get_prediction_dataframe(self, df, flow_column, days=7):
        predictions = self.predict_next_days(df, flow_column, days)
        
        if predictions is None:
            return None
        
        if 'timestamp' in df.columns:
            last_date = df['timestamp'].max()
        else:
            last_date = pd.Timestamp.now()
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        pred_df = pd.DataFrame({
            'date': future_dates,
            'predicted_usage': predictions
        })
        
        return pred_df
    
    def save_model(self):
        try:
            if self.model_type == 'random_forest':
                with open(self.rf_model_file, 'wb') as f:
                    pickle.dump(self.model, f)
            elif self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                self.model.save(self.lstm_model_file)
            
            scaler_data = {
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'lookback': self.lookback,
                'model_type': self.model_type
            }
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(scaler_data, f)
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        try:
            if os.path.exists(self.scaler_file):
                with open(self.scaler_file, 'rb') as f:
                    scaler_data = pickle.load(f)
                    self.scaler_X = scaler_data['scaler_X']
                    self.scaler_y = scaler_data['scaler_y']
                    self.lookback = scaler_data['lookback']
                    self.model_type = scaler_data['model_type']
            
            if self.model_type == 'random_forest' and os.path.exists(self.rf_model_file):
                with open(self.rf_model_file, 'rb') as f:
                    self.model = pickle.load(f)
                return True
            elif self.model_type == 'lstm' and TENSORFLOW_AVAILABLE and os.path.exists(self.lstm_model_file):
                self.model = keras.models.load_model(self.lstm_model_file)
                return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False
