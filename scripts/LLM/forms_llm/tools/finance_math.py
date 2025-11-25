# scripts/tools/finance_math.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PaymentFrequency = Literal["monthly", "yearly"]


@dataclass
class EmiInput:
    principal: float                 # loan amount
    annual_rate_percent: float       # e.g. 8.0 for 8%
    tenure_months: int               # total number of months
    frequency: PaymentFrequency = "monthly"


@dataclass
class EmiResult:
    principal: float
    annual_rate_percent: float
    tenure_months: int
    frequency: PaymentFrequency

    payment_amount: float            # EMI per period
    total_paid: float                # principal + interest
    total_interest: float            # interest over life of loan
    effective_rate_per_period: float # e.g. 0.0066 for 0.66% per month


def compute_emi(inputs: EmiInput) -> EmiResult:
    """
    Standard EMI calculation for fixed-rate loans.

    P = principal
    r = interest rate per period (e.g. per month)
    n = number of periods

    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

    If interest rate is 0, EMI = P / n.
    """
    P = float(inputs.principal)
    annual_rate = float(inputs.annual_rate_percent) / 100.0

    # Decide number of periods and periodic rate
    if inputs.frequency == "monthly":
        n = int(inputs.tenure_months)
        r = annual_rate / 12.0
    elif inputs.frequency == "yearly":
        # interpret tenure_months in years
        n = max(1, int(round(inputs.tenure_months / 12.0)))
        r = annual_rate
    else:
        raise ValueError(f"Unsupported payment frequency: {inputs.frequency}")

    if n <= 0:
        raise ValueError("tenure_months must be > 0")

    # Zero-interest special case
    if annual_rate == 0:
        payment = P / n
    else:
        factor = (1 + r) ** n
        payment = P * r * factor / (factor - 1)

    total_paid = payment * n
    total_interest = total_paid - P

    return EmiResult(
        principal=P,
        annual_rate_percent=inputs.annual_rate_percent,
        tenure_months=inputs.tenure_months,
        frequency=inputs.frequency,
        payment_amount=round(payment, 2),
        total_paid=round(total_paid, 2),
        total_interest=round(total_interest, 2),
        effective_rate_per_period=r,
    )


if __name__ == "__main__":
    # quick sanity check
    sample = EmiInput(
        principal=10000.0,
        annual_rate_percent=8.0,
        tenure_months=36,
        frequency="monthly",
    )
    res = compute_emi(sample)
    print(res)
