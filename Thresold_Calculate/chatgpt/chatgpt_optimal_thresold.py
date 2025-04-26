import pandas as pd
from sklearn.cluster import KMeans

# === Load your metrics file ===
df = pd.read_csv("query_threshold_metrics.csv")

# === Apply clustering to separate product vs non-product queries
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['top_similarity_score', 'avg_similarity_score', 'distance_to_centroid']])

# Label clusters (product = the one with higher avg top score)
cluster_means = df.groupby('cluster')['top_similarity_score'].mean()
product_cluster = cluster_means.idxmax()
df['label'] = df['cluster'].apply(lambda x: "Product" if x == product_cluster else "Non-Product")

# === Calculate recommended thresholds
top_score_thresh = df[df['label'] == "Non-Product"]['top_similarity_score'].max()
avg_score_thresh = df[df['label'] == "Non-Product"]['avg_similarity_score'].max()
centroid_thresh = df[df['label'] == "Product"]['distance_to_centroid'].max()

# === Save to txt file
thresholds = f"""
Recommended Thresholds:
------------------------
top_score_thresh     = {top_score_thresh:.4f}
avg_score_thresh     = {avg_score_thresh:.4f}
centroid_thresh      = {centroid_thresh:.4f}
"""

with open("recommended_thresholds.txt", "w") as file:
    file.write(thresholds)

print(thresholds)
print("✅ Thresholds saved to 'recommended_thresholds.txt'")
