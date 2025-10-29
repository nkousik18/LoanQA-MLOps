"""
🌍 Global Test Configuration for LoanDocQA+ Project
==================================================
Ensures consistent imports and working directories
across all test modules, whether run individually or
with coverage (`pytest --cov`).

✅ Guarantees:
    - All tests execute from project root
    - 'scripts/', 'dags/', 'data/', and 'logs/' paths resolve correctly
    - Works identically in local, CI/CD, and PyCharm runs
"""

import os
import sys
import pytest


# ============================================================
# 1️⃣ Normalize Working Directory to Project Root
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Force all tests to execute from the project root
os.chdir(ROOT_DIR)

# Ensure project root is always importable
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ============================================================
# 2️⃣ Auto-create required directories for consistency
# ============================================================
REQUIRED_DIRS = [
    "dags",
    "data",
    "data/loan_docs",
    "data/clean_texts",
    "logs",
    "logs/test_logs",
]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 3️⃣ Pytest Hooks (Optional but Helpful)
# ============================================================

def pytest_sessionstart(session):
    """📦 Log the working directory and environment at test start."""
    print(f"\n[pytest] Running tests from: {os.getcwd()}")
    print(f"[pytest] Python path includes project root: {ROOT_DIR in sys.path}")


@pytest.fixture(scope="session", autouse=True)
def ensure_root():
    """🔗 Fixture that ensures all tests use absolute paths."""
    os.chdir(ROOT_DIR)
    yield
    os.chdir(ROOT_DIR)
