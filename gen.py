import torch
import torch.nn as nn
import os

# ایجاد پوشه weights در صورت عدم وجود
if not os.path.exists('weights'):
    os.makedirs('weights')

def generate_temporal_weights():
    print("در حال تولید وزن‌های سنتتیک برای مدل زمانی (GRU)...")

    # تنظیمات ابعاد مطابق با مقاله و فایل train.py
    input_dim = 50
    hidden_dim = 256
    latent_dim = 128
    
    # تعریف لایه‌ها دقیقاً مطابق با کلاس TemporalAutoencoder
    # ۱. بخش انکودر زمانی
    gru_enc = nn.GRU(input_dim, hidden_dim, batch_first=True)
    fc_latent = nn.Linear(hidden_dim, latent_dim)
    
    # ۲. بخش دیکودر زمانی
    fc_upscale = nn.Linear(latent_dim, hidden_dim)
    gru_dec = nn.GRU(hidden_dim, input_dim, batch_first=True)

    # تجمیع پارامترها در یک state_dict
    # نام‌گذاری‌ها باید دقیقاً با اسامی متغیرها در src/models/embeddings.py یکی باشد
    state_dict = {}
    for name, param in gru_enc.named_parameters():
        state_dict[f'gru_enc.{name}'] = param
    for name, param in fc_latent.named_parameters():
        state_dict[f'fc_latent.{name}'] = param
    for name, param in fc_upscale.named_parameters():
        state_dict[f'fc_upscale.{name}'] = param
    for name, param in gru_dec.named_parameters():
        state_dict[f'gru_dec.{name}'] = param
        
    # ذخیره فایل نهایی
    torch.save(state_dict, 'weights/temporal_autoencoder.pt')
    
    # تولید وزن‌های بخش Fusion (اولویت با داده‌های چشمی طبق مقاله)
    fusion_state = {
        'weights': torch.tensor([0.1, 0.45, 0.15, 0.1, 0.2]) 
    }
    torch.save(fusion_state, 'weights/fusion_weights.pt')
    
    print("فایل‌های وزن با موفقیت ساخته شدند:")
    print("- weights/temporal_autoencoder.pt")
    print("- weights/fusion_weights.pt")

if __name__ == "__main__":
    generate_temporal_weights()