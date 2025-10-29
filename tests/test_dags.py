"""
✅ Unit tests for Airflow DAGs in the LoanDocQA+ pipeline.
Verifies DAG integrity, dependencies, and naming conventions.
"""

import os
import sys
import pytest
import importlib.util
from airflow.models import DAG

# ============================================================
# Ensure 'scripts/' is importable when running via pytest
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extraction_pipeline.config import setup_logger
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)


# ============================================================
# Logger Setup
# ============================================================
LOG_DIR = "logs/test_logs"
os.makedirs(LOG_DIR, exist_ok=True)
test_logger = setup_logger("dag_tests", log_type="test")
test_logger.info("🚀 Starting DAG validation suite.")

# ============================================================
# Helper: Dynamic DAG Importer
# ============================================================
def import_dag_file(filepath):
    """Safely import DAG Python files and return list of DAG objects."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"❌ Failed to import {filepath}: {e}")

    dags = [v for v in vars(module).values() if isinstance(v, DAG)]
    return dags

if not os.path.exists("dags"):
    pytest.skip("❌ 'dags/' directory not found — skipping DAG integrity tests.")

# ============================================================
# 1️⃣ DAG Integrity
# ============================================================
@pytest.mark.parametrize("dag_file", [
    f"dags/{f}" for f in os.listdir("dags")
    if f.endswith(".py") and not f.startswith("__")
])
def test_dag_integrity(dag_file):
    """✅ Ensure all DAGs load and contain valid structures."""
    test_logger.info(f"🧩 Validating DAG file: {dag_file}")
    dags = import_dag_file(dag_file)
    assert dags, f"❌ No DAG found in {dag_file}"

    for dag in dags:
        test_logger.info(f"🔍 Checking DAG: {dag.dag_id}")
        assert isinstance(dag.dag_id, str)
        assert len(dag.tasks) > 0, f"❌ {dag.dag_id} has no tasks."
        assert all(task.task_id for task in dag.tasks), f"❌ {dag.dag_id} contains unnamed tasks."
        assert all(hasattr(task, "execute") for task in dag.tasks), f"❌ {dag.dag_id} has tasks missing execute()."
        test_logger.info(f"✅ DAG {dag.dag_id} passed integrity checks.")


# ============================================================
# 2️⃣ DAG Dependencies
# ============================================================
def test_dag_dependencies():
    """✅ Ensure task dependencies are consistent (no broken references)."""
    for file in os.listdir("dags"):
        if not file.endswith(".py") or file.startswith("__"):
            continue

        dags = import_dag_file(f"dags/{file}")
        for dag in dags:
            test_logger.info(f"🔗 Checking dependencies in DAG: {dag.dag_id}")
            for task in dag.tasks:
                for downstream in task.downstream_task_ids:
                    assert downstream in dag.task_dict, (
                        f"❌ Downstream task '{downstream}' not found in DAG '{dag.dag_id}'"
                    )
                for upstream in task.upstream_task_ids:
                    assert upstream in dag.task_dict, (
                        f"❌ Upstream task '{upstream}' not found in DAG '{dag.dag_id}'"
                    )
            test_logger.info(f"✅ Dependencies OK for DAG {dag.dag_id}.")


# ============================================================
# 3️⃣ DAG Task Naming Conventions
# ============================================================
def test_dag_task_naming_convention():
    """✅ Validate snake_case and descriptive task_ids."""
    for file in os.listdir("dags"):
        if not file.endswith(".py") or file.startswith("__"):
            continue

        dags = import_dag_file(f"dags/{file}")
        for dag in dags:
            test_logger.info(f"✏️ Checking naming conventions for {dag.dag_id}")
            for task in dag.tasks:
                assert "_" in task.task_id, f"❌ Task '{task.task_id}' should follow snake_case."
                assert len(task.task_id) > 3, f"❌ Task '{task.task_id}' name too short."
            test_logger.info(f"✅ Naming OK for DAG {dag.dag_id}.")
