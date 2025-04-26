import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
from googletrans import Translator

# === Load your model assets ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("D:/yen/LLM/intet_search_api/model_assets/faiss_index.index")
df = pd.read_csv("D:/yen/LLM/intet_search_api/model_assets/products.csv")
translator = Translator()

# === Compute centroid of FAISS index ===
def get_centroid_vector(faiss_index):
    all_vectors = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)
    faiss_index.reconstruct_n(0, faiss_index.ntotal, all_vectors)
    centroid = np.mean(all_vectors, axis=0)
    return centroid

centroid_vector = get_centroid_vector(index)

# === Load test queries ===
input_csv = "test_queries.csv"  # path to the CSV file you downloaded
queries_df = pd.read_csv(input_csv)

# === Analyze queries ===
log = []

for query in queries_df["query"]:
    original_query = query
    try:
        lang = detect(query)
        if lang != "en":
            query = translator.translate(query, dest="en").text
    except Exception as e:
        print(f"[Error] Language detection or translation failed: {e}")

    query_embedding = np.array([embedding_model.encode(query)], dtype="float32")
    distances, indices = index.search(query_embedding, 980)
    similarities = 1 / (1 + distances[0])

    log.append({
        "original_query": original_query,
        "translated_query": query,
        "top_similarity_score": np.max(similarities),
        "avg_similarity_score": np.mean(similarities),
        "distance_to_centroid": np.linalg.norm(query_embedding - centroid_vector)
    })

# === Save the results ===
output_csv = "query_threshold_metrics.csv"
pd.DataFrame(log).to_csv(output_csv, index=False)
print(f"✅ Metrics saved to: {output_csv}")
