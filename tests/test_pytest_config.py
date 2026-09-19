from pathlib import Path


def test_pytest_discovers_split_model_bank_review_cases() -> None:
    config = Path(__file__).parents[1] / "pytest.ini"
    text = config.read_text(encoding="utf-8")
    assert "model_bank_review_cases.py" in text
