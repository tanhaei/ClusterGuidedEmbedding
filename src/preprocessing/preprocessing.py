import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

class BioArcTemporalPreprocessor:
    """
    پیش‌پردازش داده‌ها برای مدل‌سازی زمانی (Longitudinal Data).
    تبدیل رکوردهای پراکنده بیمار به توالی‌های زمانی.
    """
    def __init__(self, max_visits=10):
        self.max_visits = max_visits
        self.scaler = StandardScaler()

    def create_sequences(self, df, patient_col='PatientID', time_col='VisitDate'):
        """
        تبدیل دیتافریم به آرایه ۳ بعدی (Patients, Time, Features).
        """
        # مرتب‌سازی بر اساس زمان
        df = df.sort_values([patient_col, time_col])
        
        patients = df[patient_col].unique()
        feature_cols = [c for c in df.columns if c not in [patient_col, time_col]]
        num_features = len(feature_cols)
        
        # نرمال‌سازی ویژگی‌ها
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        sequences = []
        for pid in patients:
            p_data = df[df[patient_col] == pid][feature_cols].values
            
            # Padding یا Truncation برای یکسان‌سازی طول توالی‌ها
            if len(p_data) < self.max_visits:
                pad = np.zeros((self.max_visits - len(p_data), num_features))
                p_data = np.vstack([p_data, pad])
            else:
                p_data = p_data[:self.max_visits]
                
            sequences.append(p_data)
            
        return torch.tensor(np.array(sequences), dtype=torch.float32)