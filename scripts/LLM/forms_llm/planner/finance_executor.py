# scripts/LLM/forms_llm/planner/finance_executor.py

from __future__ import annotations

import os
import sys
from typing import List, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.LLM.forms_llm.planner.finance_tasks import (
    FinanceTask,
    FinanceTaskKind,
    FinanceTaskResult,
)
from scripts.LLM.forms_llm.tools.finance_math import EmiInput, compute_emi

# Reuse central logging infra
from scripts.aws_extraction_scripts.log_utils import get_logger

LOGGER = get_logger(__name__)


def run_finance_task(task: FinanceTask) -> FinanceTaskResult:
    """
    Execute ONE finance task using pure math (no LLM).

    Right now we only support EMI, but the pattern is generic.
    """
    LOGGER.info(
        f"[finance_executor] Running finance task: "
        f"task_id={task.id}, kind={task.kind.value}, "
        f"principal={task.principal}, rate={task.annual_rate_percent}%, "
        f"tenure_months={task.tenure_months}, frequency={task.frequency}, "
        f"currency={task.currency}"
    )

    if task.kind != FinanceTaskKind.EMI:
        msg = f"Unsupported FinanceTaskKind: {task.kind}"
        LOGGER.error(f"[finance_executor] {msg}")
        raise ValueError(msg)

    emi_input = EmiInput(
        principal=task.principal,
        annual_rate_percent=task.annual_rate_percent,
        tenure_months=task.tenure_months,
        frequency=task.frequency,  # "monthly" or "yearly"
    )
    emi_res = compute_emi(emi_input)

    LOGGER.info(
        f"[finance_executor] Completed EMI computation for task_id={task.id}: "
        f"payment_amount={emi_res.payment_amount} {task.currency}, "
        f"total_paid={emi_res.total_paid} {task.currency}, "
        f"total_interest={emi_res.total_interest} {task.currency}"
    )

    return FinanceTaskResult(
        task_id=task.id,
        kind=task.kind,
        currency=task.currency,
        payment_amount=emi_res.payment_amount,
        total_paid=emi_res.total_paid,
        total_interest=emi_res.total_interest,
        tenure_months=emi_res.tenure_months,
        annual_rate_percent=emi_res.annual_rate_percent,
        raw={
            "effective_rate_per_period": emi_res.effective_rate_per_period,
        },
    )


def run_finance_tasks(tasks: List[FinanceTask]) -> Dict[str, FinanceTaskResult]:
    """
    Run several finance tasks and return a dict keyed by task_id.
    """
    LOGGER.info(
        f"[finance_executor] Running {len(tasks)} finance tasks "
        f"(kinds={[t.kind.value for t in tasks]})"
    )
    results: Dict[str, FinanceTaskResult] = {}
    for task in tasks:
        res = run_finance_task(task)
        results[task.id] = res
    LOGGER.info(
        f"[finance_executor] Completed all finance tasks "
        f"(count={len(results)})"
    )
    return results


if __name__ == "__main__":
    # manual test with some made-up values
    sample_tasks = [
        FinanceTask(
            id="f1",
            kind=FinanceTaskKind.EMI,
            principal=10000.0,
            annual_rate_percent=8.0,
            tenure_months=36,
            frequency="monthly",
            currency="USD",
        )
    ]

    results = run_finance_tasks(sample_tasks)
    for task_id, res in results.items():
        print("\n==============================")
        print(f"Task {task_id} | kind={res.kind.value}")
        print(f"Currency: {res.currency}")
        print(f"Principal (approx from EMI*tenure): {res.payment_amount * res.tenure_months:.2f}")
        print(f"EMI per period: {res.payment_amount} {res.currency}")
        print(f"Total paid: {res.total_paid} {res.currency}")
        print(f"Total interest: {res.total_interest} {res.currency}")
