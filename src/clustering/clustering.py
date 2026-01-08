from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_feature_clustering(feature_matrix, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(feature_matrix.T) # خوشه‌بندی روی ویژگی‌ها
    
    score = silhouette_score(feature_matrix.T, clusters)
    print(f"Silhouette Score for K={k}: {score:.4f}") #
    return clusters
