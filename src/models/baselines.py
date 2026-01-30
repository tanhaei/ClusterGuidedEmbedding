import torch
import torch.nn as nn

class GlobalBaselineModel(nn.Module):
    """
    مدل Baseline که تمام ویژگی‌ها را به صورت یک بردار واحد (Flat) دریافت می‌کند[cite: 144].
    این مدل فاقد ساختار خوشه‌بندی و مکانیزم توجه (Attention) است.
    """
    def __init__(self, total_input_dim, embed_dim=128):
        super(GlobalBaselineModel, self).__init__()
        # یک شبکه عصبی ساده برای یادگیری نمایش کلی
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim) # خروجی: Global Embedding
        )

    def forward(self, x):
        """
        ورودی x شامل تمام ویژگی‌های بیمار به صورت Concatenated است.
        """
        return self.network(x)
