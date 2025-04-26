import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load your product data
df = pd.read_csv("model_assets/products.csv")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the 'model' column or whatever text you're searching over
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

# Save the embeddings as numpy array
np.save("model_assets/product_embeddings.npy", embeddings)

# Create FAISS index
embedding_dim = embeddings.shape[1]

#yen
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(np.array(embeddings).astype("float32"))

#rabib bhai
m = 8
bits = 8
nlist = 5  # number of clusters
quantizer = faiss.IndexFlatL2(embedding_dim)
index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, bits)

index.train(embeddings)  # 🔧 This is crucial
index.add(embeddings)


# Save FAISS index
faiss.write_index(index, "model_assets/faiss_index.index")

print("✅ FAISS index and embeddings saved.")


