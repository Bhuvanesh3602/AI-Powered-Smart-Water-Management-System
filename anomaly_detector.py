import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os


class AnomalyDetector:
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.model_file = "anomaly_model.pkl"
        self.feature_columns = []
        
    def prepare_features(self, df, flow_column):
        features = pd.DataFrame()
        
        features[flow_column] = df[flow_column]
        
        if 'timestamp' in df.columns:
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['day_of_month'] = df['timestamp'].dt.day
        
        features['flow_rolling_mean'] = df[flow_column].rolling(window=10, min_periods=1).mean()
        features['flow_rolling_std'] = df[flow_column].rolling(window=10, min_periods=1).std().fillna(0)
        
        features['flow_diff'] = df[flow_column].diff().fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != flow_column and col in df.columns:
                features[col] = df[col]
        
        features = features.fillna(features.mean())
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def train(self, df, flow_column):
        features = self.prepare_features(df, flow_column)
        
        features_scaled = self.scaler.fit_transform(features)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False
        )
        
        self.model.fit(features_scaled)
        
        self.save_model()
        
        return self.model
    
    def detect_anomalies(self, df, flow_column):
        if self.model is None:
            self.train(df, flow_column)
        
        features = self.prepare_features(df, flow_column)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled)
        
        anomaly_scores = self.model.score_samples(features_scaled)
        
        df_result = df.copy()
        df_result['is_anomaly'] = predictions == -1
        df_result['anomaly_score'] = anomaly_scores
        
        return df_result
    
    def get_anomaly_summary(self, df_with_anomalies, flow_column):
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == True]
        
        summary = {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(df_with_anomalies)) * 100 if len(df_with_anomalies) > 0 else 0,
            'anomaly_indices': anomalies.index.tolist(),
        }
        
        if len(anomalies) > 0:
            summary['avg_anomaly_flow'] = anomalies[flow_column].mean()
            summary['max_anomaly_flow'] = anomalies[flow_column].max()
            summary['min_anomaly_flow'] = anomalies[flow_column].min()
            
            if 'timestamp' in anomalies.columns:
                summary['first_anomaly'] = anomalies['timestamp'].min()
                summary['last_anomaly'] = anomalies['timestamp'].max()
                summary['anomaly_dates'] = anomalies['timestamp'].dt.date.unique().tolist()
        
        return summary
    
    def get_leak_alerts(self, df_with_anomalies, flow_column, threshold_multiplier=2.0):
        normal_data = df_with_anomalies[df_with_anomalies['is_anomaly'] == False]
        normal_mean = normal_data[flow_column].mean() if len(normal_data) > 0 else df_with_anomalies[flow_column].mean()
        normal_std = normal_data[flow_column].std() if len(normal_data) > 0 else df_with_anomalies[flow_column].std()
        
        leak_threshold = normal_mean + (threshold_multiplier * normal_std)
        
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == True]
        
        leaks = anomalies[anomalies[flow_column] > leak_threshold].copy()
        
        alerts = []
        for idx, row in leaks.iterrows():
            severity = 'Critical' if row[flow_column] > leak_threshold * 1.5 else 'High'
            multiplier = row[flow_column] / normal_mean if normal_mean > 0 else 1
            
            alert = {
                'index': idx,
                'timestamp': row.get('timestamp', 'Unknown'),
                'flow_rate': row[flow_column],
                'normal_flow': normal_mean,
                'multiplier': multiplier,
                'severity': severity,
                'message': f"Leak Detected - {multiplier:.1f}x normal usage"
            }
            alerts.append(alert)
        
        return alerts
    
    def save_model(self):
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'contamination': self.contamination
            }
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_columns = model_data['feature_columns']
                    self.contamination = model_data['contamination']
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False
    
    def get_anomaly_patterns(self, df_with_anomalies):
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == True]
        
        patterns = {}
        
        if 'timestamp' in anomalies.columns and len(anomalies) > 0:
            hourly_anomalies = anomalies.groupby(anomalies['timestamp'].dt.hour).size()
            patterns['peak_anomaly_hours'] = hourly_anomalies.nlargest(3).to_dict()
            
            daily_anomalies = anomalies.groupby(anomalies['timestamp'].dt.dayofweek).size()
            patterns['peak_anomaly_days'] = daily_anomalies.nlargest(3).to_dict()
        
        return patterns
