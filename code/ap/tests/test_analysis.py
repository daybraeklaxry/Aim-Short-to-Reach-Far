"""Check released aggregations and the cohort assumptions that make them valid."""
import copy
from pathlib import Path
import unittest

import numpy as np

from anchored_planning.analysis import (check_tables, local_tables, main_tables,
    main_tensor, read_csv, read_json)
from anchored_planning.action_domain import prepare_candidates, project

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(ROOT / "protocols/main.json")
        cls.records = read_json(ROOT / "results/main/successes.json")

    def test_main_reproduces_every_released_table_field(self):
        tensor = main_tensor(self.records, self.protocol)
        tables, _ = main_tables(tensor)
        self.assertEqual(check_tables(tables, ROOT / "results/main"), 160)

    def test_local_pairing_and_exact_prefix_aggregation(self):
        protocol = read_json(ROOT / "protocols/local_target.json")
        tables, summary = local_tables(read_csv(ROOT / "results/local_target/paired_queries.csv"), protocol)
        self.assertEqual(check_tables(tables, ROOT / "results/local_target"), 46)
        self.assertEqual(summary["primary_macro_effects"][1]["anchor_minus_transport_pp"], 8.59375)

    def test_duplicate_or_missing_success_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            main_tensor(self.records + [self.records[0]], self.protocol)
        with self.assertRaisesRegex(ValueError, "all 512"):
            main_tensor(self.records[:-1], self.protocol)

    def test_repeated_episode_invalidates_cluster_assumption(self):
        protocol = copy.deepcopy(self.protocol)
        rows = protocol["studies"]["cube"]["confirm"]
        rows[1]["episode"] = rows[0]["episode"]
        with self.assertRaisesRegex(ValueError, "episode clusters"):
            main_tensor(self.records, protocol)

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
