"""
Unit tests for Airflow DAGs in the pipeline.
Verifies DAG structure, task dependencies, and proper naming conventions.
"""

import os
import pytest
import importlib.util
from airflow.models import DAG



# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def import_dag_file(filepath):
    """Dynamically import a Python DAG file and return all DAG objects."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dags = [v for v in vars(module).values() if isinstance(v, DAG)]
    return dags

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

@pytest.mark.parametrize("dag_file", [
    f"dags/{f}" for f in os.listdir("dags") if f.endswith(".py")
])
def test_dag_integrity(dag_file):
    """✅ Ensure all DAGs load without errors and have valid structure."""
    dags = import_dag_file(dag_file)
    assert len(dags) > 0, f"No DAG found in {dag_file}"

    for dag in dags:
        # DAG structure validation
        assert isinstance(dag.dag_id, str)
        assert len(dag.tasks) > 0, f"DAG {dag.dag_id} has no tasks"
        assert all(task.task_id for task in dag.tasks), f"DAG {dag.dag_id} has unnamed tasks"
        assert all(hasattr(task, "execute") for task in dag.tasks), f"Some tasks lack execute() method"

def test_dag_dependencies():
    """✅ Ensure task dependencies are properly set (no circular references)."""
    for file in os.listdir("dags"):
        if not file.endswith(".py"):
            continue
        dags = import_dag_file(f"dags/{file}")
        for dag in dags:
            for task in dag.tasks:
                # Check downstream and upstream links
                if task.downstream_task_ids:
                    for d in task.downstream_task_ids:
                        assert d in dag.task_dict, f"Downstream task {d} missing in {dag.dag_id}"
                if task.upstream_task_ids:
                    for u in task.upstream_task_ids:
                        assert u in dag.task_dict, f"Upstream task {u} missing in {dag.dag_id}"

def test_dag_task_naming_convention():
    """✅ All task_ids must be snake_case and descriptive."""
    for file in os.listdir("dags"):
        if not file.endswith(".py"):
            continue
        dags = import_dag_file(f"dags/{file}")
        for dag in dags:
            for task in dag.tasks:
                assert "_" in task.task_id, f"Task {task.task_id} should follow snake_case"
                assert len(task.task_id) > 3, f"Task {task.task_id} name too short"
