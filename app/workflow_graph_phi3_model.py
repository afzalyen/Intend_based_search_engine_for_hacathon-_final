import os
import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline
from detoxify import Detoxify
from langdetect import detect
import re
from googletrans import Translator
import redis
import json
import spacy
from metrics import (
    CACHE_HIT_COUNT,
    LLM_CALL_COUNT,
    EMBEDDING_LATENCY,
    FAISS_LATENCY,
    RESULT_COUNT_SUMMARY,
    RESULT_RELEVANCE_SCORE,
    TRANSLATION_COUNT,
    BANGLISH_COUNT,
    UNSUPPORTED_LANG_COUNT,
)

# === Step 1: Load Product Data and Vector Index === #
df = pd.read_csv("model_assets/products.csv")
index = faiss.read_index("model_assets/faiss_index.index")

# === Step 2: Load Sentence Embedding Model === #
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Translate to English Model
translator = Translator()


redis_client = redis.Redis(
    host='localhost',
    port=6379,
    # password='redispassword',  # If you set Redis password
    # decode_responses=False        # Store embeddings as bytes
)

# Load SpaCy model once
nlp = spacy.load("en_core_web_sm")


flag_LLM =0
LLM_str = ""

# Extract price relation
def extract_relations(query):
    print(query)
    doc = nlp(query)
    relations = {
        'above': ['above', 'over', 'more than', 'greater than'],
        'below': ['below', 'under', 'less than', 'upto']
    }
    results = []
    
    for token in doc:
        if token.text.isdigit():
            value = float(token.text)
            found_relation = None
            
            # Check surrounding words (3 tokens before & after)
            start = max(0, token.i - 3)
            end = min(len(doc), token.i + 4)
            window = doc[start:end]
            
            # Look for relation words BEFORE the number (e.g., "under 50000")
            for i in range(0, token.i - start):
                word = window[i].text.lower()
                for rel_type, rel_words in relations.items():
                    if word in rel_words:
                        found_relation = rel_type
                        break
            
            # If no relation before, check AFTER (e.g., "50000 rupees")
            if not found_relation:
                for i in range(token.i - start + 1, len(window)):
                    word = window[i].text.lower()
                    for rel_type, rel_words in relations.items():
                        if word in rel_words:
                            found_relation = rel_type
                            break
            
            if found_relation:
                results.append({"relation": found_relation, "value": value})
            
    
    # Return the highest value if multiple numbers exist
    if results:
        return max(results, key=lambda x: x["value"])
    return None

# Apply numeric filter on results
def filter_by_price(results_df, relation_data):
    if relation_data:
        value = float(relation_data["value"])

        if relation_data["relation"] == "above":
            return results_df[results_df["price"] > value]
        elif relation_data["relation"] == "below":
            return results_df[results_df["price"] < value]
        
        
    return results_df



def semantic_cache(query: str, embedding_model) -> dict | None:
    query_embedding = embedding_model.encode([query])[0].astype(np.float32)

    with EMBEDDING_LATENCY.labels(source="cache").time():
        embedding = np.array([embedding_model.encode(query)], dtype=np.float32)

    for key in redis_client.scan_iter("q:*"):
        cached_embedding = np.frombuffer(redis_client.hget(key, "embedding"), dtype=np.float32)
        similarity = np.dot(query_embedding, cached_embedding)
        print("redis similarity score: ",similarity)
        if similarity > 0.7:
            print(f"✅ Cache Hit: {key.decode()}")
            CACHE_HIT_COUNT.inc()
            cached_result = json.loads(redis_client.hget(key, "products"))
            return cached_result

    return None

# def store_in_cache(query: str, embedding_model, products: dict):
#     query_embedding = embedding_model.encode([query])[0].astype(np.float32)
#     cache_key = f"q:{query[:40]}"  # Optional: add hash or limit length
#     redis_client.hset(cache_key, mapping={
#         "embedding": query_embedding.tobytes(),
#         "products": json.dumps(products)
#     })
#     redis_client.expire(cache_key, 60 * 60 * 24)  # Optional: 1-day TTL

def store_in_cache(query: str, embedding_model, products: dict):
    try:
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        cache_key = f"q:{query[:40]}"  # Optional: add hash or limit length
        redis_client.hset(cache_key, mapping={
            "embedding": query_embedding.tobytes(),
            "products": json.dumps(products)
        })
        redis_client.expire(cache_key, 60 * 60 * 24)  # Optional: 1-day TTL
        print(f"✅ Successfully cached query: {query}")
    except Exception as e:
        print(f"❌ Failed to cache query '{query}': {str(e)}")


# === Step 3: Load or Quantize Phi-3 Mini Model === #
def load_or_quantize_model(model_path: str, quantized_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if os.path.exists(os.path.join(quantized_path, "config.json")):
        print("✅ Loading quantized model from disk...")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_path,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        print("⚙️ Quantizing and saving model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.save_pretrained(quantized_path)
        tokenizer.save_pretrained(quantized_path)
        print("✅ Quantized model saved.")
    
    return tokenizer, model

def create_pipeline(tokenizer, model):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    )


def input_guardrails(query: str) -> dict:
    # Rule-based guardrails
    if not query or len(query.strip()) == 0:
        return {"blocked": True, "reason": " Empty input is not allowed."}
    
    if len(query) > 1000:
        return {"blocked": True, "reason": " Input is too long. Please be concise."}
    
    if re.search(r'[{}<>$&|;`]', query):  # Potential command/code injection characters
        return {"blocked": True, "reason": " Unsafe characters detected in input."}

    # Language detection
    # try:
    #     lang = detect(query)
    #     print(f"Detected Language: {lang}")
    # except:
    #     lang = "unknown"

    # Classifier-based (toxicity, etc.)
    detox_result = Detoxify('original').predict(query)
    print("Detoxify Scores:", detox_result)

    # Define a list of risk factors with their thresholds
    risk_factors = {
        'toxicity': 0.4,
        'insult': 0.4,
        'sexual_explicit': 0.4,
        'threat': 0.4
    }

    # Check each risk factor safely
    for key, threshold in risk_factors.items():
        if key in detox_result and detox_result[key] > threshold:
            return {
                "blocked": True,
                "reason": f"{key.replace('_', ' ').capitalize()} content detected."
            }

    return {"blocked": False, "reason": "Passed"}




# ✅ Function to convert Bangla digits to English digits
def convert_bangla_digits_to_english(text):
    bangla_to_english_digits = {
        '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
        '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
    }
    return ''.join(bangla_to_english_digits.get(char, char) for char in text)


# ✅ Main query processing function
# def translate_query(user_input):
#     try:
#         lang = detect(user_input)
#         print(f"[DEBUG] Detected Language: {lang}")

#         if lang == "en":
#             return {"output": user_input, "language": "English"}

#         elif lang == "bn":
#             print("[DEBUG] Translating Bangla to English...")
#             user_input = convert_bangla_digits_to_english(user_input)  # Convert digits before translation
#             translated = translator.translate(user_input, src='bn', dest='en').text
#             return {"output": translated, "language": "Bangla"}

#         else:
#             print("[DEBUG] Treating as non-English/Bangla input (possibly Banglish or other)...")
#             user_input = convert_bangla_digits_to_english(user_input)  # Convert digits before translation
#             translated = translator.translate(user_input, src='bn', dest='en').text
#             return {"output": translated, "language": "Bangla"}
#             # return {"output": user_input, "language": "Banglish"}

#     except Exception as e:
#         print(f"[ERROR] Language detection failed: {str(e)}")
#         return {"output": user_input, "language": "Unknown"}
    
# ✅ Main query processing function
def translate_query(user_input):
    try:
        user_input = convert_bangla_digits_to_english(user_input)  # Convert Bangla digits first
        translated = translator.translate(user_input, dest='en').text

        TRANSLATION_COUNT.inc()
        return {"output": translated,  "language": "Bangla"}
    except Exception as e:
        print(f"[ERROR] Translation failed: {str(e)}")
        return {"output": user_input, "language": "Bangla"}


def output_guardrail(search_string, documents_df, embedding_model, centroid_vector,
                     top_score_thresh=0.49, avg_score_thresh=0.35,
                     centroid_thresh=1.2, sharpness_thresh=0.08,
                     ):

    top_score = documents_df["similarity_score"].max()
    avg_score = documents_df["similarity_score"].mean()
    min_score = documents_df["similarity_score"].min()
    sharpness_score = top_score - min_score

    print(f"top_score: {top_score}")
    print(f"avg_score: {avg_score}")
    print(f"min_score: {min_score}")
    print(f"sharpness_score: {sharpness_score:.4f}")

    query_embedding = np.array([embedding_model.encode(search_string)], dtype="float32")
    distance = np.linalg.norm(query_embedding - centroid_vector)
    print(f"[DEBUG] Centroid Distance: {distance:.3f}")

    # Step 1: Hard filters
    if top_score < top_score_thresh:
        return {
            "blocked": True,
            "reason": "Top similarity score is too low.",
            "filtered": pd.DataFrame()
        }

    if avg_score < avg_score_thresh:
        return {
            "blocked": True,
            "reason": "Average similarity score is too low.",
            "filtered": pd.DataFrame()
        }

    if distance > centroid_thresh:
        return {
            "blocked": True,
            "reason": f"Query appears out-of-domain (distance={distance:.2f}).",
            "filtered": pd.DataFrame()
        }


    # Step 2: Relative score filtering
    relative_thresh = top_score * 0.93
    filtered = documents_df[documents_df["similarity_score"] >= relative_thresh]
    filter_ratio = len(filtered) / len(documents_df)
    print(f"[DEBUG] Filtered result count: {len(filtered)} of {len(documents_df)} ({filter_ratio*100:.2f}%)")

    # Step 3: Dynamically raise filter strictness if too much survived
    if filter_ratio > 0.75 and top_score < 0.49:
        print("[DEBUG] Too many weak results passed filter. Raising threshold.")
        relative_thresh = top_score * 0.97
        filtered = documents_df[documents_df["similarity_score"] >= relative_thresh]
        filter_ratio = len(filtered) / len(documents_df)
        print(f"[DEBUG] After re-filtering: {len(filtered)} of {len(documents_df)} ({filter_ratio*100:.2f}%)")

    # Step 4: Combine filter ratio and sharpness as confidence signal
    if sharpness_score < sharpness_thresh and filter_ratio > 0.75 and top_score < 0.49:
        return {
            "blocked": True,
            "reason": "Low sharpness and high filter pass rate suggest irrelevant query.",
            "filtered": pd.DataFrame()
        }

    if filtered.empty:
        return {
            "blocked": True,
            "reason": "None of the retrieved products are relevant enough.",
            "filtered": pd.DataFrame()
        }

    return {
        "blocked": False,
        "filtered": filtered
    }




def get_centroid_vector(faiss_index):
    all_vectors = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)
    faiss_index.reconstruct_n(0, faiss_index.ntotal, all_vectors)
    centroid = np.mean(all_vectors, axis=0)
    return centroid

CENTROID_VECTOR = get_centroid_vector(index)




# Paths
base_model_path = "D:/yen/LLM/intet_search_api/models/phi-3-mini-4k-instruct"
quantized_path = "D:/yen/LLM/intet_search_api/models/phi-3-mini-quantized"

# Load model
tokenizer, phi3_model = load_or_quantize_model(base_model_path, quantized_path)
pipe = create_pipeline(tokenizer, phi3_model)
llm = HuggingFacePipeline(pipeline=pipe)

# === Step 4: Prompt Template === #
prompt_template = PromptTemplate(
template="""
You are an AI assistant that converts user requests into short, general-purpose search queries for any product or service.
Instructions:
1. Translate the query to English if it's written in another language (e.g., Bangla, Banglish).
2. Understand what the user is **truly looking for**. Focus on **intent**, not emotion or story.
3. Return a concise **English search phrase** that works across online stores or search engines.
4. Avoid personal context or slang. Be clear, general, and product-focused.
5. Keep brand names if the user clearly mentions one (e.g., iPhone, Samsung, Nike).
6. If the query includes **sensitive or restricted items** (e.g., alcohol, adult content, weapons, narcotics).
7. Do not return explanations or translations — only the final search phrase.
Examples:
- "ami showoff korar jonno ekta iPhone kinte chai" → "luxury-looking iPhone"
- "ami fashion show e jabo, stylish dress dorkar" → "stylish outfits for fashion events"
- "আমি একটা সস্তা ব্যাকপ্যাক খুঁজছি কলেজের জন্য" → "affordable backpacks for college"
- "I want shoes that are comfortable but look classy" → "comfortable classy shoes"
- "I'm looking for a chair for long hours of study" → "ergonomic study chairs"
- "কোন ভালো quality alcohol দাও" → "high quality alcohol [sensitive]"
- "where to buy adult toys in Dhaka" → "adult toys in Dhaka [sensitive]"
- "homemade gun for self defense" → "homemade gun for self defense [sensitive]"
Query:
{query}
Optimized Search Phrase:
""",
input_variables=["query"]
)

retrieval_grader = prompt_template | llm | StrOutputParser()

# === Step 5: Graph Components === #
class GraphState(TypedDict):
    question: str
    search_string: str
    score: np.float32
    documents: pd.DataFrame
    products: dict

def generate_search_string(state: GraphState) -> dict:
    global LLM_str
    question = state["question"]
    global flag_LLM
    # if user's query after translation is less than 100 char then bypass LLM
    LLM_CALL_COUNT.inc()
    if len(question)<100:
        flag_LLM = 1
        user_english_input = question
        return {
        "search_string": user_english_input,
        "question": question
    }
    LLM_str=question
    user_english_input = retrieval_grader.invoke({"query": question})

    print("🔎 Before augmenting: ", user_english_input)
    #user_translated_english_input= user_english_input
    match = re.search(r'"(.*?)"', user_english_input, re.DOTALL)
    extracted = user_english_input
    if match:
        extracted = match.group(1)
        print("Extracted:", extracted)
    else:
        print("No match found.")
    user_english_input = extracted.strip()
    print("🔎 Optimized Search: ", user_english_input)

    

    #guardrail after LLM
    guardrail_result = input_guardrails(user_english_input)
    print('guardrail_result: ',guardrail_result)
    if guardrail_result["blocked"]:
        print("Input Blocked by Guardrails:", guardrail_result["reason"])
        return {"error": guardrail_result["reason"]}
    return {
        "search_string": user_english_input,
        "question": question
    }

def retrieve(state: GraphState) -> dict:
    print("📡 Retrieving Top Matches...")
    search_string = state["search_string"]

    with EMBEDDING_LATENCY.labels(source="search").time():
        query_embedding = np.array([embedding_model.encode(search_string)], dtype="float32")

    with FAISS_LATENCY.time():
        distances, indices = index.search(query_embedding, 10)


    query_embedding = np.array([embedding_model.encode(search_string)], dtype="float32")
    distances, indices = index.search(query_embedding, 200)

    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 / (1 + distances[0])
    RESULT_COUNT_SUMMARY.observe(len(results))
    RESULT_RELEVANCE_SCORE.observe(float(results["similarity_score"].mean()))

    
    return {
        "search_string": search_string,
        "question": state["question"],
        "score": results["similarity_score"],
        "documents": results
    }

# def output_func(state: GraphState) -> dict:
#     results = state["documents"]
#     products = {}
#     for _, row in results.iterrows():
#         model_name = row.get("model", "Unknown")
#         products[model_name] = {
#             "price": row.get("price"),
#             "similarity_score": float(row["similarity_score"])
#         }

#     return {
#         "question": state["question"],
#         "search_string": state["search_string"],
#         "score": state["score"],
#         "documents": results,
#         "products": products
#     }

def output_func(state: GraphState) -> dict:
    results = state["documents"]
    search_string = state["search_string"]

    # Apply output guardrail
    output_guardrail_result = output_guardrail(
        search_string=search_string,
        documents_df=results,
        embedding_model=embedding_model,
        centroid_vector=CENTROID_VECTOR
    )
    print(output_guardrail_result,"output_guardrail_result")

    if output_guardrail_result["blocked"]:
        print("🚫 Output Blocked:", output_guardrail_result["reason"])
    
        return {
        "question": state["question"],
        "search_string": search_string,
        "score": results["similarity_score"],
        "documents": results,
        "products": output_guardrail_result["reason"]
    }

    filtered_results = output_guardrail_result["filtered"]

    # Extract relation like 'under 50000' #Rabib Bhai
    # relation_data = extract_relations(search_string)
    # print('relation_data: ',relation_data)


    # Filter by price relation
    # filtered_results = filter_by_price(filtered_results, relation_data)





    print("len of filtered_results :",len(filtered_results))

    # Proceed with filtered results
    products = {}
    for _, row in filtered_results.iterrows():
        model_name = row.get("model", "Unknown")
        products[model_name] = {
            # "price": row.get("price"),
            "similarity_score": float(row["similarity_score"])
        }

    return {
        "question": state["question"],
        "search_string": search_string,
        "score": filtered_results["similarity_score"],
        "documents": filtered_results,
        "products": products
    }

# def printer(state: GraphState) -> dict:
#     print("\n Final Product Matches:")
#     for product, details in state["products"].items():
#         print(f" {product} - ${details['price']} (Score: {details['similarity_score']:.3f})")
#     return state

# === Step 6: Graph Definition === #
workflow = StateGraph(GraphState)
workflow.add_node("generate_search_string", generate_search_string)
workflow.add_node("retrieve", retrieve)
workflow.add_node("output_func", output_func)
# workflow.add_node("printer", printer)

workflow.set_entry_point("generate_search_string")
workflow.add_edge("generate_search_string", "retrieve")
workflow.add_edge("retrieve", "output_func")
workflow.add_edge("output_func", END)
# workflow.add_edge("printer", END)

app = workflow.compile()

# === Step 7: Main Execution Function === #
# def run_workflow_phi3_mini(query: str) -> dict:
#     # Step 1: Guardrails
#     print('query: ',query)
#     query = query.strip()
#     #query translation
#     transale_result = translate_query(query)
#     print(f"✅ Output: {transale_result['output']}, language: {transale_result['language']}\n")

#     translated_query = transale_result['output']


#     #guardrail before LLM
#     input_guardrail_result = input_guardrails(translated_query)
#     print('guardrail_result: ',input_guardrail_result)
#     if input_guardrail_result["blocked"]:
#         print("Input Blocked by Guardrails:", input_guardrail_result["reason"])
#         return {"error": input_guardrail_result["reason"]}

#     # Step 2: Workflow Execution
#     inputs = {"question": translated_query}
#     final_result = {}

#     for output in app.stream(inputs):
#         for key, value in output.items():
#             final_result.update(value)

#     print("Workflow Complete")
#     return final_result.get("products", {})



def run_workflow_phi3_mini(query: str) -> dict:
    global flag_LLM , LLM_str
    LLM_str = ""
    flag_LLM = 0
    print("📝 User Input:", query)
    query = query.strip()

    # Step 1: Translate Input
    trans_result = translate_query(query)
    translated_query = trans_result['output']
    print(f"🌐 Translated to English: {translated_query} [lang: {trans_result['language']}]")

    # Step 2: Guardrail before semantic embedding
    input_guardrail_result = input_guardrails(translated_query)
    if input_guardrail_result["blocked"]:
        print("🚫 Blocked (input guardrail):", input_guardrail_result["reason"])
        return {"error": input_guardrail_result["reason"]}

    # Step 3: Redis Semantic Cache Check
    cached_result = semantic_cache(translated_query, embedding_model)
    if cached_result:
        print("📦 Redis Cache Hit ✅")
        
        # Optional: Cache the exact raw input too, if desired
        #store_in_cache(query, embedding_model, cached_result)
        
        return cached_result

    # Step 4: Run LangGraph → which internally applies LLM (if needed) & FAISS retrieval
    print("⚙️ Running LangGraph Workflow...")
    final_result = {}
    inputs = {"question": translated_query}

    for output in app.stream(inputs):
        for key, value in output.items():
            final_result.update(value)

    print("✅ Workflow Complete")

    products = final_result.get("products", {})
    query_used_for_embedding = final_result.get("search_string", translated_query)
    
    print("query_used_for_embedding: ",query_used_for_embedding)
    # Step 5: Store the final query + result in Redis
    print(flag_LLM)
    if flag_LLM==1:

        print("LLM_str: ",LLM_str, "query_used_for_embedding: ", query_used_for_embedding)
        store_in_cache(LLM_str, embedding_model, products)

    store_in_cache(query_used_for_embedding, embedding_model, products)
    

    return products

# === Example Run === #
# products = run_workflow_phi3_mini("ami fashion show e jabo, stylish dress dorkar")
# for model, details in products.items():
#     print(f"{model} - ${details['price']} (Score: {details['similarity_score']})")
