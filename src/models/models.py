import torch
import torch.nn as nn

class NumericAutoencoder(nn.Module):
    """
    مدل اتوانکودر برای ویژگی‌های عددی.
    ساختار لایه‌ها دقیقاً با فایل وزن تولید شده هماهنگ است.
    """
    def __init__(self, input_dim=50, latent_dim=128):
        super(NumericAutoencoder, self).__init__()
        # ساختار Sequential برای مطابقت با فایل وزن تولید شده
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

class CategoricalEmbedder(nn.Module):
    """
    مدل تعبیه‌سازی برای کدهای تشخیصی و دارو (Med2Vec-style).
    """
    def __init__(self, vocab_size=1000, embed_dim=128):
        super(CategoricalEmbedder, self).__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        
    def forward(self, x):
        return torch.relu(self.embedding(x))