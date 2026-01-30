from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd

def run_preliminary_clustering(feature_matrix, k_range=[3, 5, 7, 10]):
    results = []
    print("--- نتایج اعتبارسنجی خوشه‌بندی روی MIMIC-IV ---")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        
    
        sil = silhouette_score(feature_matrix, labels)
        ch = calinski_harabasz_score(feature_matrix, labels)
        
        results.append({'K': k, 'Silhouette': sil, 'CH_Index': ch})
        print(f"K={k} | Silhouette Score: {sil:.4f} | CH Index: {ch:.2f}")
        
    return pd.DataFrame(results)
