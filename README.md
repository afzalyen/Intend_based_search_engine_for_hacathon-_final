# Intend_based_search_engine_for_hacathon-_final

📦 Intent-based Product Search API This project implements a semantic product search engine that intelligently understands user intent and retrieves the most relevant products from a FAISS-based vector store.

It uses:

Sentence Transformers for semantic embedding

FAISS for approximate nearest neighbor search

HuggingFace Transformers (Phi-3 Mini) for query rewriting

Redis for semantic caching

Detoxify for input safety

Google Translate API for language translation

SpaCy for price relation extraction

🏗 Project Structure

Component Description products.csv Product dataset (text + price info) faiss_index.index FAISS index of product embeddings embedding_model SentenceTransformer model for text embeddings phi-3-mini-4k-instruct HuggingFace LLM for query rewriting redis Caching previously seen queries & results detoxify Guardrails against unsafe/toxic user input translate_query() Handles Bangla/Banglish to English conversion output_guardrail() Filters bad retrievals (low similarity, off-domain) LangGraph Defines and executes the query-to-retrieval pipeline 🚀 How It Works User Input Handling:

Translate non-English queries (Bangla, Banglish) into English.

Check for unsafe/toxic/invalid inputs (guardrails).

Semantic Caching:

Check Redis cache: if a similar query was asked before, return cached results immediately.

Query Optimization:

If the query is long, refine it using the Phi-3 Mini LLM to generate a clean search intent.

Vector Retrieval:

Embed the final query and search FAISS index for top 200 closest products.

Output Filtering:

Remove irrelevant results using sharpness, centroid distance, and average similarity scores.

Result Storage:

Store new query results in Redis for faster future retrieval.

Final Response:

Return matching product models and their similarity scores.

🛠 Requirements Python 3.9+

Redis Server

HuggingFace Transformers

LangChain

Detoxify

FAISS

SpaCy (with en_core_web_sm)

Googletrans

Sentence-Transformers

Torch

Pandas

LangGraph

Install dependencies:

bash Copy Edit pip install -r requirements.txt ⚡ Example python Copy Edit products = run_workflow_phi3_mini("আমি কলেজের জন্য সস্তা ল্যাপটপ খুঁজছি")

for model, details in products.items(): print(f"{model} - Score: {details['similarity_score']:.3f}") 🧠 Features ✅ Bangla/Banglish/English query understanding

✅ Toxic input detection (Detoxify)

✅ Query rewriting using Phi-3 Mini

✅ Fast retrieval with FAISS

✅ Smart output filtering for relevance

✅ Semantic caching in Redis

✅ Handles price filtering (e.g., "under 50000")

🔥 Future Improvements Add more languages (multilingual support)

Auto-refresh FAISS index with new products

Use faster, smaller LLM for rewriting

Add UI for searching

✨ Credits OpenAI, HuggingFace, LangChain, FAISS, Detoxify, Google Translate
