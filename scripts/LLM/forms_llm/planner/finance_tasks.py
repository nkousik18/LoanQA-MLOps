# scripts/planner/finance_tasks.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class FinanceTaskKind(str, Enum):
    EMI = "finance_emi"   # later you can add more kinds if needed


@dataclass
class FinanceTask:
    """
    A pure numeric finance task.

    Later, principal/rate/tenure can come from doc_qa tasks or user input.
    For now you can pass them manually to test the tool.
    """
    id: str
    kind: FinanceTaskKind

    principal: float
    annual_rate_percent: float
    tenure_months: int
    frequency: str = "monthly"  # "monthly" or "yearly"
    currency: str = "USD"
    domain: str = "finance"     # just a label, matches your planner idea


@dataclass
class FinanceTaskResult:
    task_id: str
    kind: FinanceTaskKind
    currency: str

    payment_amount: float      # EMI
    total_paid: float          # principal + interest
    total_interest: float      # interest over loan life
    tenure_months: int
    annual_rate_percent: float

    # raw extra fields (for debugging / advanced use later)
    raw: Dict[str, Any]
