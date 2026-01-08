import torch
import torch.optim as optim
from src.preprocessing.preprocessor import BioArcTemporalPreprocessor
from src.models.embeddings import TemporalAutoencoder
from src.fusion.integration import WeightedFusion

def train_temporal_model():
    print("--- Training Temporal Cluster-Guided Framework ---")
    
    # تنظیمات ابعاد (مثلاً ۱۰ بازدید، هر کدام ۵۰ ویژگی)
    max_visits = 10
    input_dim = 50
    latent_dim = 128
    
    # ۱. تولید داده‌های زمانی فرضی (Batch, Time, Features)
    # در واقعیت از BioArcTemporalPreprocessor استفاده می‌شود
    dummy_data = torch.randn(100, max_visits, input_dim)
    
    # ۲. مقداردهی مدل‌های زمانی
    model = TemporalAutoencoder(input_dim, latent_dim=latent_dim)
    fusion = WeightedFusion(num_clusters=5)
    
    optimizer = optim.Adam(list(model.parameters()) + list(fusion.parameters()), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # ۳. حلقه آموزش
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        latent, recon = model(dummy_data)
        
        # محاسبه Loss بازسازی زمانی
        loss = criterion(recon, dummy_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Reconstruction Loss: {loss.item():.4f}")
            
    # ۴. ذخیره وزن‌ها
    torch.save(model.state_dict(), 'weights/temporal_autoencoder.pt')
    torch.save(fusion.state_dict(), 'weights/fusion_weights.pt')
    print("Temporal weights saved.")

if __name__ == "__main__":
    train_temporal_model()