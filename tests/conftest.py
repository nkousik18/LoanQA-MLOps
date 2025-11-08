"""
Global test configuration for LoanDocQA+ project.
Ensures consistent imports and working directories across all test modules.
"""

import os
import sys
import pytest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.chdir(ROOT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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


def pytest_sessionstart(session):
    """Log the working directory and environment at test start."""
    print(f"\n[pytest] Running tests from: {os.getcwd()}")
    print(f"[pytest] Python path includes project root: {ROOT_DIR in sys.path}")


@pytest.fixture(scope="session", autouse=True)
def ensure_root():
    """Fixture that ensures all tests use absolute paths."""
    os.chdir(ROOT_DIR)
    yield
    os.chdir(ROOT_DIR)