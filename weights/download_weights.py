import torch

# ذخیره وزن‌های یک مدل (مثلاً Autoencoder برای داده‌های عددی) [cite: 101]
def save_cluster_weights(model, cluster_name):
    path = f"weights/{cluster_name}_weights.pt"
    torch.save(model.state_dict(), path)
    print(f"Weights for {cluster_name} saved successfully.")

# بارگذاری وزن‌ها در سیستم مقصد
def load_cluster_weights(model, cluster_name):
    model.load_state_dict(torch.load(f"weights/{cluster_name}_weights.pt"))
    model.eval()
    return model
