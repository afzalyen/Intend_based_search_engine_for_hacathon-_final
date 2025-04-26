import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Load your existing setup
df = pd.read_csv("D:/yen/LLM/intet_search_api/model_assets/products.csv")
index = faiss.read_index("D:/yen/LLM/intet_search_api/model_assets/faiss_index.index")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Calculate centroid vector
def get_centroid_vector(faiss_index):
    all_vectors = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)
    faiss_index.reconstruct_n(0, faiss_index.ntotal, all_vectors)
    centroid = np.mean(all_vectors, axis=0)
    return centroid

CENTROID_VECTOR = get_centroid_vector(index)

# Function to analyze a query
def analyze_query(query, embedding_model, index, centroid_vector, df):
    # Get embedding
    query_embedding = np.array([embedding_model.encode(query)], dtype="float32")
    
    # Search index
    distances, indices = index.search(query_embedding, 980)
    
    # Get results
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 / (1 + distances[0])
    
    # Calculate metrics
    top_score = results["similarity_score"].max()
    avg_score = results["similarity_score"].mean()
    centroid_distance = np.linalg.norm(query_embedding - centroid_vector)
    
    return {
        'query': query,
        'top_score': top_score,
        'avg_score': avg_score,
        'centroid_distance': centroid_distance,
        'num_results': len(results[results["similarity_score"] > 0.3])  # Count of somewhat relevant results
    }

# Main analysis function
def analyze_thresholds(queries_file, output_file):
    # Load test queries
    test_queries = pd.read_csv(queries_file)
    
    results = []
    
    for i, row in test_queries.iterrows():
        query = row['query']
        query_type = row['type']
        
        print(f"Processing {i+1}/{len(test_queries)}: {query}")
        
        analysis = analyze_query(query, embedding_model, index, CENTROID_VECTOR, df)
        analysis['type'] = query_type
        results.append(analysis)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")
    
    return results_df

# Run the analysis
if __name__ == "__main__":
    analyze_thresholds('deep_seek_test_queries.csv', 'threshold_analysis_results.csv')