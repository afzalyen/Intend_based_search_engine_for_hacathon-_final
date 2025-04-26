from prometheus_client import Counter, Summary

API_HIT_COUNT = Counter("api_request_count", "Total number of API requests")
API_SUCCESS_COUNT = Counter("api_success_count", "Number of successful responses")
API_ERROR_COUNT = Counter("api_error_count", "Number of error responses")
API_LATENCY = Summary("api_request_latency_seconds", "Time taken to process the request")

QUERY_HIT_COUNT = Counter("query_hit_count", "Number of queries received")
CACHE_HIT_COUNT = Counter("cache_hit_count", "Number of queries served from cache")
LLM_CALL_COUNT = Counter("llm_call_count", "LLM model call count")
RESULT_COUNT_SUMMARY = Summary("search_result_count", "Number of results returned")
RESULT_RELEVANCE_SCORE = Summary("search_result_relevance", "Relevance score of top search results")

EMBEDDING_LATENCY = Summary(
    "embedding_time_seconds",
    "Time taken to generate query embedding",
    ["source"]  # ← label key
)
FAISS_LATENCY = Summary("faiss_search_time_seconds", "Time taken for FAISS vector search")

TRANSLATION_COUNT = Counter("query_translation_count", "Queries translated to English")
BANGLISH_COUNT = Counter("banglish_detected_count", "Banglish queries detected")
UNSUPPORTED_LANG_COUNT = Counter("unsupported_language_count", "Unsupported language detected")

from prometheus_client import Summary
