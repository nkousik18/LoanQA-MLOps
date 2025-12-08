# monitoring/exporters/llm_metrics_exporter.py

from prometheus_client import Counter, Histogram

# ------------------------------
# METRIC DEFINITIONS
# ------------------------------

REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of LLM microservice requests",
    ["mode"]  # summary / translation / explanation / finance / chat
)

RETRIEVAL_SCORE = Histogram(
    "retrieval_topk_scores",
    "Histogram of top-k retrieval scores",
    buckets=[0.2, 0.4, 0.6, 0.8, 1.0]
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_ms",
    "Latency of retrieval stage",
    buckets=[10, 30, 50, 100, 200, 500, 1000]
)

ROUTER_CONFIDENCE = Histogram(
    "router_confidence_scores",
    "Confidence distribution of intent router",
    buckets=[0.2, 0.4, 0.6, 0.8, 1.0]
)

LLM_LATENCY = Histogram(
    "llm_generation_latency_ms",
    "Latency of LLM generation stage",
    buckets=[50, 100, 200, 500, 1000, 2000]
)

TOTAL_LATENCY = Histogram(
    "llm_total_latency_ms",
    "Total request latency",
    buckets=[100, 200, 500, 1000, 2000, 5000]
)



# ------------------------------
# METRIC UPDATE FUNCTION
# ------------------------------

def update_metrics(timing: dict, router_confidence: float):
    # Count request
    REQUEST_COUNT.labels("llm_request").inc()

    # Router confidence
    ROUTER_CONFIDENCE.observe(router_confidence)

    # Latency metrics
    RETRIEVAL_LATENCY.observe(timing["retrieval_ms"])
    LLM_LATENCY.observe(timing["llm_ms"])
    TOTAL_LATENCY.observe(timing["total_ms"])
