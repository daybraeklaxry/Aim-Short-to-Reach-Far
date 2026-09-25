"""Check memory separation and action clipping."""
from pathlib import Path
import unittest

import numpy as np

from anchored_planning.analysis import read_json
from anchored_planning.action_domain import prepare_candidates, project

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def test_query_episodes_are_absent_from_retrieval_memory(self):
        for study in ("main", "local_target"):
            protocol = read_json(ROOT / "protocols" / f"{study}.json")
            for task, spec in protocol["studies"].items():
                train = set(protocol["bank_train_episodes"][task])
                self.assertFalse(train & {row["episode"] for row in spec["confirm"]})


class ActionDomainTests(unittest.TestCase):
    def test_projection_is_used_before_normalized_scoring(self):
        class AffineScaler:
            def transform(self, values):
                return (values - np.array([0.25, -0.25])) / 2

        raw = np.array([[[2.0, -2.0], [0.5, -0.5]]], dtype=np.float32)
        effective, normalized, counts = prepare_candidates(raw, [-1, -1], [1, 1], AffineScaler())
        np.testing.assert_array_equal(effective, [[[1, -1], [0.5, -0.5]]])
        np.testing.assert_array_equal(normalized, [[[0.375, -0.375], [0.125, -0.125]]])
        self.assertEqual(counts, dict(chunks=1, action_steps=1, coordinates=2))
        np.testing.assert_array_equal(raw, [[[2, -2], [0.5, -0.5]]])

    def test_clean_prefix_has_zero_projection_counts(self):
        effective, counts = project(np.empty((0, 2)), [-1, -1], [1, 1])
        self.assertEqual(effective.shape, (0, 2))
        self.assertEqual(counts, dict(chunks=0, action_steps=0, coordinates=0))


if __name__ == "__main__":
    unittest.main()
