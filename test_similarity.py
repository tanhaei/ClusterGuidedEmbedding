import torch
import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.models.embeddings import TemporalAutoencoder
from src.fusion.integration import WeightedFusion

def calculate_mrr(y_true, y_scores):
    """
    محاسبه میانگین رتبه معکوس (Mean Reciprocal Rank).
    برای ارزیابی کیفیت رتبه‌بندی بیماران مشابه استفاده می‌شود.
    """
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = np.take(y_true, order)
    rs = (np.where(y_true_sorted == 1)[0]) + 1
    if rs.size:
        return 1.0 / rs[0]
    return 0

def evaluate_similarity():
    print("--- ارزیابی شباهت کلینیکی (مدل زمانی) ---")
    
    # تنظیمات مسیر فایل‌های وزن تولید شده در مرحله قبلی
    weights_dir = 'weights'
    temporal_path = os.path.join(weights_dir, 'temporal_autoencoder.pt')
    fusion_path = os.path.join(weights_dir, 'fusion_weights.pt')

    # ۱. بررسی وجود فایل‌های مورد نیاز
    if not os.path.exists(temporal_path) or not os.path.exists(fusion_path):
        print("خطا: فایل‌های وزن یافت نشدند. ابتدا generate_weights.py را اجرا کنید.")
        return

    # ۲. مقداردهی مدل‌های زمانی (GRU-based)
    # ابعاد: ۵۰ ویژگی در هر ویزیت، فضای نهفته ۱۲۸
    input_dim = 50
    max_visits = 10
    temporal_model = TemporalAutoencoder(input_dim=input_dim, latent_dim=128)
    fusion_layer = WeightedFusion(num_clusters=5)

    try:
        # بارگذاری وزن‌های مدل GRU
        temporal_model.load_state_dict(torch.load(temporal_path))
        
        # بارگذاری وزن‌های لایه ادغام (اولویت‌های بالینی)
        fusion_state = torch.load(fusion_path)
        if 'weights' in fusion_state:
            fusion_layer.weights.data = fusion_state['weights']
        
        temporal_model.eval()
        fusion_layer.eval()
        print("مدل‌های زمانی با موفقیت بارگذاری شدند.")
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        return

    # ۳. شبیه‌سازی داده‌های تست (۵۰۰ جفت بیمار با توالی زمانی)
    # ساختار داده: (تعداد بیمار، تعداد ویزیت، تعداد ویژگی)
    num_test = 500
    test_sequences = torch.randn(num_test, max_visits, input_dim)
    
    # برچسب‌های فرضی مشابهت (استاندارد طلایی)
    y_true = np.random.randint(0, 2, num_test) 

    # ۴. استخراج نمایش یکپارچه و محاسبه زمان پاسخ‌دهی (Inference Time)
    start_time = time.perf_counter()
    with torch.no_grad():
        # استخراج بردارهای نهفته از توالی‌های زمانی
        latent_vecs, _ = temporal_model(test_sequences)
        
        # شبیه‌سازی ادغام ۵ خوشه (چشمی، دموگرافیک و ...)
        unified_repr = fusion_layer([latent_vecs] * 5)
        
    # محاسبه شباهت کسینوسی بین بیمار هدف (Query) و سایرین
    query_vector = unified_repr[0].unsqueeze(0)
    similarity_scores = torch.cosine_similarity(query_vector, unified_repr).numpy()
    
    inference_ms = (time.perf_counter() - start_time) * 1000

    # ۵. محاسبه و چاپ معیارهای نهایی مطابق با نتایج مقاله
    y_pred = (similarity_scores > 0.7).astype(int)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, similarity_scores)
    mrr = calculate_mrr(y_true, similarity_scores)

    print(f"\n--- نتایج ارزیابی نهایی (Temporal Framework) ---")
    print(f"Precision: {p:.2f}")
    print(f"Recall:    {r:.2f}")
    print(f"F1-Score:  {f1:.2f}")
    print(f"AUC-ROC:   {auc:.2f}")
    print(f"MRR:       {mrr:.2f}")
    print(f"زمان استنتاج: {inference_ms/num_test:.2f} میلی‌ثانیه به ازای هر بیمار")

if __name__ == "__main__":
    evaluate_similarity()