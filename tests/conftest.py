import pytest

@pytest.fixture(scope="session")
def setup_env(tmp_path_factory):
    """Global test setup fixture."""
    base_dir = tmp_path_factory.mktemp("loan_test_env")
    return {"base_dir": base_dir}

