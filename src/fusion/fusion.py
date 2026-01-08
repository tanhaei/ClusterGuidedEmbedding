import torch
import torch.nn as nn

class WeightedFusion(nn.Module):
    """
    لایه‌ی ادغام وزن‌دار برای ترکیب خوشه‌های مختلف بالینی.
    وزن‌ها نشان‌دهنده اهمیت هر خوشه در شباهت نهایی بیمار هستند.
    """
    def __init__(self, num_clusters=5):
        super(WeightedFusion, self).__init__()
        # وزن‌ها به صورت پارامتر یادگیری‌شونده تعریف شده‌اند
        self.weights = nn.Parameter(torch.ones(num_clusters))
        
    def forward(self, embeddings_list):
        # اعمال وزن‌های اختصاصی به هر خوشه
        weighted_embeddings = []
        for i, emb in enumerate(embeddings_list):
            weighted_embeddings.append(emb * self.weights[i])
            
        # الحاق (Concatenate) بردارها برای ساخت نمایش یکپارچه
        return torch.cat(weighted_embeddings, dim=-1)