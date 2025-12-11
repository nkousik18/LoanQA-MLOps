"""
cloud_monitoring.py
-------------------
Google Cloud Monitoring integration for Doc-Understand.

Tracks:
1. PDF processing times and success rates
2. Query response times and success rates
3. RAG retrieval quality scores
4. System errors and active sessions
5. User activity patterns

Industry-standard monitoring following Google SRE practices.
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# Ensure project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from google.cloud import monitoring_v3
    from google.cloud import logging as cloud_logging
    CLOUD_MONITORING_AVAILABLE = True
except ImportError:
    CLOUD_MONITORING_AVAILABLE = False
    print("⚠️  google-cloud-monitoring not installed. Install with:")
    print("   pip install google-cloud-monitoring google-cloud-logging")

from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)

# Your GCP Project ID
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "doc-understand")


class CloudMonitor:
    """
    Google Cloud Monitoring client for Doc-Understand.
    
    Sends custom metrics to Cloud Monitoring for real-time dashboards
    and alerting.
    """
    
    def __init__(self, project_id: str = PROJECT_ID):
        """
        Initialize Cloud Monitoring client.
        
        Args:
            project_id: GCP project ID (default: doc-understand)
        """
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        
        if not CLOUD_MONITORING_AVAILABLE:
            LOGGER.warning(
                "Cloud Monitoring not available. Metrics will only be logged locally."
            )
            self.client = None
            self.logging_client = None
            return
        
        try:
            self.client = monitoring_v3.MetricServiceClient()
            self.logging_client = cloud_logging.Client(project=project_id)
            LOGGER.info(f"✅ Cloud Monitoring initialized for project: {project_id}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Cloud Monitoring: {e}")
            self.client = None
            self.logging_client = None
    
    def _write_time_series(
        self,
        metric_type: str,
        value: float,
        metric_labels: Optional[Dict[str, str]] = None,
        value_type: str = "DOUBLE",
    ):
        """
        Write a time series data point to Cloud Monitoring.
        
        Args:
            metric_type: Metric type (e.g., 'pdf_processing_time')
            value: Metric value
            metric_labels: Optional labels for filtering
            value_type: DOUBLE, INT64, or BOOL
        """
        if self.client is None:
            LOGGER.debug(
                f"[LOCAL ONLY] Metric: {metric_type}={value}, labels={metric_labels}"
            )
            return
        
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"custom.googleapis.com/doc_understand/{metric_type}"
            
            # Add labels
            if metric_labels:
                for key, val in metric_labels.items():
                    series.metric.labels[key] = str(val)
            
            # Resource (generic_task for custom metrics)
            series.resource.type = "global"
            
            # Create data point
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10 ** 9)
            interval = monitoring_v3.TimeInterval(
                {"end_time": {"seconds": seconds, "nanos": nanos}}
            )
            point = monitoring_v3.Point(
                {"interval": interval, "value": {value_type.lower() + "_value": value}}
            )
            series.points = [point]
            
            # Write to Cloud Monitoring
            self.client.create_time_series(
                name=self.project_name, time_series=[series]
            )
            
            LOGGER.debug(f"✅ Sent metric to Cloud Monitoring: {metric_type}={value}")
            
        except Exception as e:
            LOGGER.error(f"Failed to write metric {metric_type}: {e}")
    
    def _log_event(
        self,
        message: str,
        severity: str = "INFO",
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Log an event to Cloud Logging.
        
        Args:
            message: Log message
            severity: DEBUG, INFO, WARNING, ERROR, CRITICAL
            labels: Optional labels for filtering
        """
        if self.logging_client is None:
            LOGGER.info(f"[LOCAL LOG] {severity}: {message}")
            return
        
        try:
            logger = self.logging_client.logger("doc-understand")
            logger.log_text(message, severity=severity, labels=labels or {})
        except Exception as e:
            LOGGER.error(f"Failed to write log: {e}")
    
    # =========================================================================
    # Public Methods: Log Specific Events
    # =========================================================================
    
    def log_pdf_processing(
        self,
        session_id: str,
        user_id: str,
        pdf_name: str,
        duration_seconds: float,
        success: bool,
        error_msg: Optional[str] = None,
    ):
        """
        Log PDF processing event.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            pdf_name: Name of PDF processed
            duration_seconds: Time taken to process
            success: Whether processing succeeded
            error_msg: Error message if failed
        """
        LOGGER.info(
            f"📄 PDF Processing: session={session_id}, user={user_id}, "
            f"duration={duration_seconds:.2f}s, success={success}"
        )
        
        # Metric 1: Processing time
        self._write_time_series(
            metric_type="pdf_processing_time",
            value=duration_seconds,
            metric_labels={
                "user_id": user_id,
                "success": str(success),
            },
        )
        
        # Metric 2: Processing count
        self._write_time_series(
            metric_type="pdf_processed_count",
            value=1,
            metric_labels={
                "user_id": user_id,
                "success": str(success),
            },
            value_type="INT64",
        )
        
        # Metric 3: Success rate (1 or 0)
        self._write_time_series(
            metric_type="pdf_success_rate",
            value=1.0 if success else 0.0,
            metric_labels={"user_id": user_id},
        )
        
        # Log event
        severity = "INFO" if success else "ERROR"
        message = (
            f"PDF processed: {pdf_name} in {duration_seconds:.2f}s "
            f"(success={success})"
        )
        if error_msg:
            message += f" | Error: {error_msg}"
        
        self._log_event(
            message,
            severity=severity,
            labels={
                "session_id": session_id,
                "user_id": user_id,
                "event_type": "pdf_processing",
            },
        )
    
    def log_query_execution(
        self,
        session_id: str,
        user_id: str,
        query: str,
        duration_seconds: float,
        success: bool,
        query_type: Optional[str] = None,
        rag_avg_score: Optional[float] = None,
        error_msg: Optional[str] = None,
    ):
        """
        Log query execution event.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            query: User query (truncated for privacy)
            duration_seconds: Time taken to respond
            success: Whether query succeeded
            query_type: Type (doc_qa, doc_explain, etc.)
            rag_avg_score: Average RAG retrieval score
            error_msg: Error message if failed
        """
        LOGGER.info(
            f"💬 Query: session={session_id}, user={user_id}, "
            f"type={query_type}, duration={duration_seconds:.2f}s, success={success}"
        )
        
        # Metric 1: Query response time
        self._write_time_series(
            metric_type="query_response_time",
            value=duration_seconds,
            metric_labels={
                "user_id": user_id,
                "query_type": query_type or "unknown",
                "success": str(success),
            },
        )
        
        # Metric 2: Query count
        self._write_time_series(
            metric_type="query_count",
            value=1,
            metric_labels={
                "user_id": user_id,
                "query_type": query_type or "unknown",
                "success": str(success),
            },
            value_type="INT64",
        )
        
        # Metric 3: Query success rate
        self._write_time_series(
            metric_type="query_success_rate",
            value=1.0 if success else 0.0,
            metric_labels={
                "user_id": user_id,
                "query_type": query_type or "unknown",
            },
        )
        
        # Metric 4: RAG quality score (if available)
        if rag_avg_score is not None:
            self._write_time_series(
                metric_type="rag_quality_score",
                value=rag_avg_score,
                metric_labels={
                    "user_id": user_id,
                    "session_id": session_id,
                },
            )
        
        # Log event
        severity = "INFO" if success else "ERROR"
        query_preview = query[:100] + "..." if len(query) > 100 else query
        message = (
            f"Query executed: '{query_preview}' in {duration_seconds:.2f}s "
            f"(success={success})"
        )
        if rag_avg_score:
            message += f" | RAG score: {rag_avg_score:.3f}"
        if error_msg:
            message += f" | Error: {error_msg}"
        
        self._log_event(
            message,
            severity=severity,
            labels={
                "session_id": session_id,
                "user_id": user_id,
                "query_type": query_type or "unknown",
                "event_type": "query_execution",
            },
        )
    
    def log_rag_retrieval(
        self,
        session_id: str,
        query: str,
        num_chunks: int,
        avg_score: float,
        top_score: float,
        duration_seconds: float,
    ):
        """
        Log RAG retrieval quality metrics.
        
        Args:
            session_id: Session identifier
            query: User query
            num_chunks: Number of chunks retrieved
            avg_score: Average similarity score
            top_score: Highest similarity score
            duration_seconds: Retrieval time
        """
        LOGGER.info(
            f"🔍 RAG Retrieval: session={session_id}, chunks={num_chunks}, "
            f"avg_score={avg_score:.3f}, top_score={top_score:.3f}"
        )
        
        # Metric 1: Average RAG score
        self._write_time_series(
            metric_type="rag_avg_similarity",
            value=avg_score,
            metric_labels={"session_id": session_id},
        )
        
        # Metric 2: Top RAG score
        self._write_time_series(
            metric_type="rag_top_similarity",
            value=top_score,
            metric_labels={"session_id": session_id},
        )
        
        # Metric 3: RAG retrieval time
        self._write_time_series(
            metric_type="rag_retrieval_time",
            value=duration_seconds,
            metric_labels={"session_id": session_id},
        )
        
        # Metric 4: Chunks retrieved
        self._write_time_series(
            metric_type="rag_chunks_retrieved",
            value=num_chunks,
            metric_labels={"session_id": session_id},
            value_type="INT64",
        )
        
        # Log event
        query_preview = query[:100] + "..." if len(query) > 100 else query
        self._log_event(
            f"RAG retrieval: '{query_preview}' | "
            f"chunks={num_chunks}, avg_score={avg_score:.3f}",
            severity="INFO",
            labels={
                "session_id": session_id,
                "event_type": "rag_retrieval",
            },
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an error event.
        
        Args:
            error_type: Type of error (textract_failed, rag_failed, etc.)
            error_message: Error description
            session_id: Optional session identifier
            user_id: Optional user identifier
            context: Optional additional context
        """
        LOGGER.error(f"❌ Error: type={error_type}, message={error_message}")
        
        # Metric: Error count
        self._write_time_series(
            metric_type="error_count",
            value=1,
            metric_labels={
                "error_type": error_type,
                "user_id": user_id or "unknown",
            },
            value_type="INT64",
        )
        
        # Log event
        labels = {
            "error_type": error_type,
            "event_type": "error",
        }
        if session_id:
            labels["session_id"] = session_id
        if user_id:
            labels["user_id"] = user_id
        
        message = f"Error [{error_type}]: {error_message}"
        if context:
            message += f" | Context: {context}"
        
        self._log_event(message, severity="ERROR", labels=labels)
    
    def log_active_session(
        self,
        session_id: str,
        user_id: str,
        active: bool = True,
    ):
        """
        Log active session count.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            active: True if session is active, False if closing
        """
        # Metric: Active sessions (gauge)
        self._write_time_series(
            metric_type="active_sessions",
            value=1 if active else 0,
            metric_labels={
                "session_id": session_id,
                "user_id": user_id,
            },
            value_type="INT64",
        )


# =============================================================================
# Global singleton instance
# =============================================================================
_monitor_instance: Optional[CloudMonitor] = None


def get_monitor() -> CloudMonitor:
    """
    Get or create the global CloudMonitor instance.
    
    Returns:
        CloudMonitor instance (singleton)
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CloudMonitor(project_id=PROJECT_ID)
    return _monitor_instance


# =============================================================================
# Convenience functions for easy use
# =============================================================================

def log_pdf_processing(
    session_id: str,
    user_id: str,
    pdf_name: str,
    duration_seconds: float,
    success: bool,
    error_msg: Optional[str] = None,
):
    """Convenience wrapper for logging PDF processing."""
    monitor = get_monitor()
    monitor.log_pdf_processing(
        session_id, user_id, pdf_name, duration_seconds, success, error_msg
    )


def log_query_execution(
    session_id: str,
    user_id: str,
    query: str,
    duration_seconds: float,
    success: bool,
    query_type: Optional[str] = None,
    rag_avg_score: Optional[float] = None,
    error_msg: Optional[str] = None,
):
    """Convenience wrapper for logging query execution."""
    monitor = get_monitor()
    monitor.log_query_execution(
        session_id,
        user_id,
        query,
        duration_seconds,
        success,
        query_type,
        rag_avg_score,
        error_msg,
    )


def log_rag_retrieval(
    session_id: str,
    query: str,
    num_chunks: int,
    avg_score: float,
    top_score: float,
    duration_seconds: float,
):
    """Convenience wrapper for logging RAG retrieval."""
    monitor = get_monitor()
    monitor.log_rag_retrieval(
        session_id, query, num_chunks, avg_score, top_score, duration_seconds
    )


def log_error(
    error_type: str,
    error_message: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """Convenience wrapper for logging errors."""
    monitor = get_monitor()
    monitor.log_error(error_type, error_message, session_id, user_id, context)


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    print("Testing Cloud Monitoring Integration...\n")
    
    monitor = get_monitor()
    
    # Test 1: PDF Processing
    print("1. Testing PDF processing metrics...")
    monitor.log_pdf_processing(
        session_id="test_session_001",
        user_id="test_user",
        pdf_name="test_loan.pdf",
        duration_seconds=45.3,
        success=True,
    )
    
    # Test 2: Query Execution
    print("2. Testing query execution metrics...")
    monitor.log_query_execution(
        session_id="test_session_001",
        user_id="test_user",
        query="What is the interest rate in this loan agreement?",
        duration_seconds=3.2,
        success=True,
        query_type="doc_qa",
        rag_avg_score=0.85,
    )
    
    # Test 3: RAG Retrieval
    print("3. Testing RAG retrieval metrics...")
    monitor.log_rag_retrieval(
        session_id="test_session_001",
        query="What is the repayment schedule?",
        num_chunks=8,
        avg_score=0.78,
        top_score=0.92,
        duration_seconds=0.5,
    )
    
    # Test 4: Error
    print("4. Testing error logging...")
    monitor.log_error(
        error_type="textract_timeout",
        error_message="Textract job timed out after 300 seconds",
        session_id="test_session_002",
        user_id="test_user",
        context={"pdf": "large_document.pdf", "size_mb": 25},
    )
    
    print("\n✅ Test metrics sent to Cloud Monitoring!")
    print(f"📊 View metrics at: https://console.cloud.google.com/monitoring?project={PROJECT_ID}")
    print("   (It may take 1-2 minutes for metrics to appear)")
