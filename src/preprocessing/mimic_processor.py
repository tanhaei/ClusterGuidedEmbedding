import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class MIMICIVProcessor:
    """
    پردازشگر اختصاصی برای دیتاست MIMIC-IV جهت اعتبارسنجی خارجی.
    ویژگی‌ها را به ۵ خوشه بالینی تقسیم می‌کند.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        # تعریف نگاشت ویژگی‌ها به خوشه‌ها بر اساس مقاله
        self.cluster_definitions = {
            'demographics': ['age', 'gender', 'admission_type', 'first_careunit'],
            'vitals': ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 'sbp'],
            'labs': ['glucose', 'creatinine', 'hemoglobin', 'wbc', 'platelets'],
            'codes': ['icd_code', 'medication_name', 'procedure_code'],
            'notes': ['clinical_note_text']
        }

    def process_patient_batch(self, df):
        """
        پاکسازی و خوشه‌بندی ویژگی‌های بیماران ICU.
        """
        # حذف رکوردهای تکراری و مدیریت مقادیر گم‌شده [cite: 63, 69]
        df = df.drop_duplicates().fillna(method='ffill')
        
        # نرمال‌سازی ویژگی‌های عددی (علائم حیاتی و آزمایشگاه) [cite: 64, 76]
        numeric_cols = self.cluster_definitions['vitals'] + self.cluster_definitions['labs']
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df

    def get_cluster_data(self, df, cluster_name):
        """بازگرداندن داده‌های مربوط به یک خوشه خاص."""
        cols = self.cluster_definitions.get(cluster_name, [])
        return df[df.columns.intersection(cols)]
