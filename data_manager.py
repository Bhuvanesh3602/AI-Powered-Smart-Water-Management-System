import os
import pandas as pd
import numpy as np
import kagglehub
import pickle
from datetime import datetime, timedelta
import streamlit as st


class DataManager:
    def __init__(self):
        self.dataset_path = None
        self.df = None
        self.labels_file = "water_data_labels.pkl"
        self.user_labels = {}
        
    def download_dataset(self):
        try:
            path = kagglehub.dataset_download("talha97s/smart-water-leak-detection-dataset")
            self.dataset_path = path
            return path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_dataset(self):
        if self.dataset_path is None:
            self.dataset_path = self.download_dataset()
        
        if self.dataset_path is None:
            return None
        
        try:
            csv_files = []
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                print("No CSV files found in the dataset directory")
                return None
            
            print(f"Found CSV file: {csv_files[0]}")
            df = pd.read_csv(csv_files[0])
            
            df.columns = df.columns.str.strip()
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 'date' in df.columns or 'Date' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'Date'
                df['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
            
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.df = df
            self.load_user_labels()
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def get_dataframe(self):
        if self.df is None:
            self.load_dataset()
        return self.df
    
    def add_label(self, index: int, label: str):
        self.user_labels[index] = {
            'label': label,
            'timestamp': datetime.now().isoformat()
        }
        self.save_user_labels()
    
    def remove_label(self, index: int):
        if index in self.user_labels:
            del self.user_labels[index]
            self.save_user_labels()
    
    def get_label(self, index: int):
        return self.user_labels.get(index, {}).get('label', None)
    
    def save_user_labels(self):
        try:
            with open(self.labels_file, 'wb') as f:
                pickle.dump(self.user_labels, f)
        except Exception as e:
            print(f"Error saving labels: {e}")
    
    def load_user_labels(self):
        try:
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'rb') as f:
                    self.user_labels = pickle.load(f)
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.user_labels = {}
    
    def get_usage_statistics(self):
        if self.df is None:
            return {}
        
        flow_col = self._get_flow_column()
        if flow_col is None:
            return {}
        
        stats = {
            'total_records': len(self.df),
            'avg_flow_rate': self.df[flow_col].mean(),
            'max_flow_rate': self.df[flow_col].max(),
            'min_flow_rate': self.df[flow_col].min(),
            'std_flow_rate': self.df[flow_col].std(),
        }
        
        if 'timestamp' in self.df.columns:
            stats['date_range'] = f"{self.df['timestamp'].min()} to {self.df['timestamp'].max()}"
            
            self.df['date'] = self.df['timestamp'].dt.date
            daily_usage = self.df.groupby('date')[flow_col].sum()
            stats['avg_daily_usage'] = daily_usage.mean()
            stats['total_usage'] = self.df[flow_col].sum()
        
        return stats
    
    def _get_flow_column(self):
        possible_flow_cols = ['flow_rate', 'Flow_Rate', 'flow', 'Flow', 'water_flow', 'consumption']
        for col in possible_flow_cols:
            if col in self.df.columns:
                return col
        
        if len(self.df.select_dtypes(include=[np.number]).columns) > 0:
            return self.df.select_dtypes(include=[np.number]).columns[0]
        
        return None
    
    def get_daily_usage(self):
        if self.df is None or 'timestamp' not in self.df.columns:
            return pd.DataFrame()
        
        flow_col = self._get_flow_column()
        if flow_col is None:
            return pd.DataFrame()
        
        self.df['date'] = self.df['timestamp'].dt.date
        daily_usage = self.df.groupby('date')[flow_col].agg(['sum', 'mean', 'max']).reset_index()
        daily_usage.columns = ['date', 'total_usage', 'avg_usage', 'max_usage']
        
        return daily_usage
    
    def get_hourly_pattern(self):
        if self.df is None or 'timestamp' not in self.df.columns:
            return pd.DataFrame()
        
        flow_col = self._get_flow_column()
        if flow_col is None:
            return pd.DataFrame()
        
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly_pattern = self.df.groupby('hour')[flow_col].mean().reset_index()
        hourly_pattern.columns = ['hour', 'avg_flow_rate']
        
        return hourly_pattern
    
    def export_labeled_data(self):
        if self.df is None:
            return None
        
        df_export = self.df.copy()
        df_export['user_label'] = df_export.index.map(lambda x: self.get_label(x))
        
        return df_export
