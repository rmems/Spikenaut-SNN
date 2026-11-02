"""pytest discovery config guard.

A ``unittest.TestCase`` rather than a bare pytest function: pytest collects
both, but ``python -m unittest tests.test_pytest_config`` ran 0 tests from the
bare function -- and on Python 3.12+ unittest exits 5 ("NO TESTS RAN") when
nothing runs.
"""

from __future__ import annotations

import unittest
from pathlib import Path


class PytestConfigTests(unittest.TestCase):
    def test_pytest_discovers_split_model_bank_review_cases(self) -> None:
        config = Path(__file__).parents[1] / "pytest.ini"
        text = config.read_text(encoding="utf-8")
        self.assertIn("model_bank_review_cases.py", text)


if __name__ == "__main__":
    unittest.main()
