import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from transformers import AutoTokenizer

class BioArcPreprocessor:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.scaler = StandardScaler()
        self.mlb_diag = MultiLabelBinarizer()
        self.mlb_med = MultiLabelBinarizer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def process_numeric(self, df_numeric):
        # مدیریت داده‌های مفقود با میانگین طبق نظر داور
        df_filled = df_numeric.fillna(df_numeric.median())
        return self.scaler.fit_transform(df_filled)
    
    def process_categorical(self, diagnosis_list, med_list):
        # تبدیل کدهای ICD و داروها به بردار [cite: 81, 106]
        diag_enc = self.mlb_diag.fit_transform(diagnosis_list)
        med_enc = self.mlb_med.fit_transform(med_list)
        return np.hstack([diag_enc, med_enc])
    
    def tokenize_notes(self, notes_list):
        # توکن‌سازی یادداشت‌های بالینی برای ClinicalBERT [cite: 82, 113]
        return self.tokenizer(notes_list, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
