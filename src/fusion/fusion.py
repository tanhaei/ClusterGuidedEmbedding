import torch

class WeightedFusion(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_clusters))
        
    def forward(self, embeddings_list):
        # embeddings_list: لیست بردارهای تعبیه از خوشه‌های مختلف
        weighted_embeddings = []
        for i, emb in enumerate(embeddings_list):
            weighted_embeddings.append(emb * self.weights[i])
            
        return torch.cat(weighted_embeddings, dim=-1)
