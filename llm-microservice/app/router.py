import logging
import time
from typing import Dict, Any

# ================================
# MLFLOW EVALUATION IMPORTS
# ================================
from tracking.retrieval_eval import run_retrieval_experiment
from tracking.router_eval import run_router_experiment
from tracking.llm_eval import run_llm_experiment
from tracking.end_to_end_eval import run_end_to_end_experiment


# ================================
# METRICS + APP LOGIC IMPORTS
# ================================
from monitoring.exporters.llm_metrics_exporter import update_metrics
from app.prompt_router import build_prompt
from app.output_validation import OutputValidator
from app.backend import generate, chat
from app.retriever import retriever

logger = logging.getLogger(__name__)
validator = OutputValidator()

DEFAULT_TOP_K = 3


async def route(payload: Dict[str, Any]) -> Dict[str, Any]:

    # ================================
    # CHAT HANDLING (NO MLFLOW)
    # ================================
    if "messages" in payload:
        msgs = payload["messages"]
        logger.info("[Router] Chat request detected")
        llm_output = await chat(msgs)
        return {
            "mode": "chat",
            "router_confidence": 1.0,
            "response": llm_output
        }

    # ================================
    # QUESTION EXTRACT
    # ================================
    question = payload.get("question", "").strip()
    context = payload.get("context", "")
    top_k = payload.get("top_k", DEFAULT_TOP_K)
    explicit_mode = payload.get("mode")

    if not question:
        return {"error": "Missing 'question' field."}

    logger.info(f"[Router] Question: {question}")

    # ================================
      # TIMING START
    # ================================
    t_start = time.time()

    # --- Retrieval timing ---
    t_r1 = time.time()
    retrieved_chunks = []
    retrieved_text = ""

    if context:
        retrieved_text = context
    else:
        try:
            retrieved_chunks = retriever.retrieve(question, top_k=top_k)
            retrieved_text = "\n\n".join(c["text"] for c in retrieved_chunks)
        except Exception as e:
            logger.error(f"[Router] Retriever failed: {e}")

    t_r2 = time.time()

    # ================================
    # MLFLOW: RETRIEVAL LOGGING
    # ================================
    try:
        run_retrieval_experiment(
            question=question,
            retrieved_chunks=retrieved_chunks,
            retrieval_time_ms=(t_r2 - t_r1) * 1000
        )
    except Exception as e:
        logger.warning(f"[MLflow] Retrieval logging failed: {e}")

    # ================================
    # ROUTER + PROMPT BUILD
    # ================================
    t_p1 = time.time()
    prompt, mode, conf = build_prompt(
        question=question,
        context_docs=retrieved_text,
        retrieved_chunks=retrieved_chunks,
        mode=explicit_mode
    )
    logger.info(f"[Router] Prompt built | mode={mode} | ctx_len={len(retrieved_text)}")
    t_p2 = time.time()

    # ================================
    # MLFLOW: ROUTER LOGGING
    # ================================
    try:
        run_router_experiment(
            question=question,
            predicted_mode=mode,
            confidence=conf,
            router_time_ms=(t_p2 - t_p1) * 1000
        )
    except Exception as e:
        logger.warning(f"[MLflow] Router logging failed: {e}")

    # ================================
    # LLM CALL
    # ================================
    t_l1 = time.time()
    llm_output = await generate(prompt)
    t_l2 = time.time()

    # ================================
    # VALIDATION
    # ================================
    validation = validator.validate(llm_output, mode, retrieved_text, retrieved_chunks)
    if not validation["valid"]:
        logger.warning(f"[Validator] {validation['issues']}")
        llm_output = validator.rewrite(llm_output, mode, retrieved_text)

    # ================================
    # MLFLOW: LLM LOGGING
    # ================================
    try:
        run_llm_experiment(
            prompt=prompt,
            llm_output=llm_output,
            validation=validation,
            llm_time_ms=(t_l2 - t_l1) * 1000
        )
    except Exception as e:
        logger.warning(f"[MLflow] LLM logging failed: {e}")

    t_end = time.time()

    # ================================
    # TIMING DICT
    # ================================
    timing = {
        "retrieval_ms": (t_r2 - t_r1) * 1000,
        "router_ms": (t_p2 - t_p1) * 1000,
        "llm_ms": (t_l2 - t_l1) * 1000,
        "total_ms": (t_end - t_start) * 1000
    }

    # ================================
    # PROMETHEUS METRIC EXPORT
    # ================================
    try:
        update_metrics(timing, conf)
    except Exception as e:
        logger.warning(f"[Metrics] Failed to update Prometheus metrics: {e}")

    # ================================
    # MLFLOW: END-TO-END LOGGING
    # ================================
    try:
        run_end_to_end_experiment(
            question=question,
            retrieved_chunks=retrieved_chunks,
            intent=mode,
            router_conf=conf,
            prompt=prompt,
            llm_output=llm_output,
            timing=timing
        )
    except Exception as e:
        logger.warning(f"[MLflow] End-to-end logging failed: {e}")

    # ================================
    # FINAL RESPONSE
    # ================================
    return {
        "mode": mode,
        "router_confidence": conf,
        "retrieval_confidence": (
            retrieved_chunks[0]["score"] if retrieved_chunks else None
        ),
        "response": llm_output,
        "retrieved_chunks": retrieved_chunks,
        "timing": timing,
    }
