import hashlib
import unittest

from tools.m1_arm_plan import build_plan
from tools.m1_cohort import FIXTURE_JSONL, FIXTURE_SPLIT, preflight
from tools.m1_features import paired_rows


class PairedFeaturesTest(unittest.TestCase):
    def test_rows_match_plan_and_keep_sensor_units(self):
        cohort = preflight(FIXTURE_JSONL, FIXTURE_SPLIT, None)
        rows = list(paired_rows(cohort))
        coverage = build_plan(cohort)["paired_coverage"]
        for split, plan in coverage.items():
            selected = [r.sample for r in rows if r.split == split]
            self.assertEqual(len(selected), plan["common_rows"])
            identity = "".join(f"{s.episode_id},{s.source_line}\n" for s in selected)
            self.assertEqual("sha256:" + hashlib.sha256(identity.encode()).hexdigest(),
                             plan["common_rows_sha256"])
        first = rows[0]
        readings = cohort.measurements[:8]
        self.assertEqual(first.raw, readings[-1])
        for value, index in zip(first.kinetic, (2, 1), strict=True):
            self.assertAlmostEqual(value, sum(r[index] for r in readings) / 8)
        self.assertEqual(first.arms()["HYBRID"], first.raw + first.kinetic)

    def test_dropout_discards_old_window(self):
        from dataclasses import replace
        cohort = preflight(FIXTURE_JSONL, FIXTURE_SPLIT, None)
        cohort.samples[8] = replace(cohort.samples[8], missing=("gpu_temp_c",))
        rows = list(paired_rows(cohort))
        self.assertEqual([r.sample.source_line for r in rows[:2]], [8, 17])
        self.assertAlmostEqual(rows[1].kinetic[0], sum(v[2] for v in cohort.measurements[9:17]) / 8)
