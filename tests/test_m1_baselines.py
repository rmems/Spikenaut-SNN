import unittest
from dataclasses import replace

from tools.m1_baselines import evaluate
from tools.m1_cohort import FIXTURE_JSONL, FIXTURE_SPLIT, preflight
from tools.q88_core import ParseError
from tools.recurrence_core import readout_target


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.cohort = preflight(FIXTURE_JSONL, FIXTURE_SPLIT, None)

    def test_persistence_uses_previous_tick_and_counts_common_rows(self):
        result = evaluate(self.cohort)
        samples = self.cohort.samples
        # Validation is one contiguous 32-row episode. The eighth tick is
        # the first common one and can use the preceding target.
        val = [s for s in samples if self.cohort.split_by_episode[s.episode_id] == "val"]
        targets = [readout_target(s.stim) for s in val]
        mae = sum(abs(targets[i][0] - targets[i-1][0]) for i in range(7, len(val))) / (len(val)-7)
        self.assertAlmostEqual(result["metrics"]["val"]["persistence"]["pooled_mae"][0], mae)
        for metrics in result["metrics"]["test"].values():
            self.assertEqual(sum(v["n_ticks"] for v in metrics["sessions"].values()), 50)

    def test_holdout_extreme_cannot_change_fitted_models(self):
        original = evaluate(self.cohort)
        self.cohort.measurements[-1] = (10000,) * 5
        changed = evaluate(self.cohort)
        self.assertEqual(original["models"], changed["models"])
        self.assertNotEqual(original["metrics"]["test"]["RAW"], changed["metrics"]["test"]["RAW"])

    def test_empty_common_split_is_refused(self):
        for i, sample in enumerate(self.cohort.samples):
            if self.cohort.split_by_episode[sample.episode_id] == "val":
                self.cohort.samples[i] = replace(sample, missing=("gpu_temp_c",))
        with self.assertRaisesRegex(ParseError, "no paired rows"):
            evaluate(self.cohort)
