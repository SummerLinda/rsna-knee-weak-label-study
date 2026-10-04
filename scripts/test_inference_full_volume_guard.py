"""CPU release-gate regressions; no competition data or CUDA required."""
import unittest
import numpy as np
from inference_full_volume_guard import (
    StrictCohortAudit, cn2_full_volume_centers, assert_prediction_vector,
)


class GuardTests(unittest.TestCase):
    def complete(self):
        audit = StrictCohortAudit(['u1', 'u2'], ['m1', 'm2'])
        for u in audit.study_ids:
            audit.record_preparation(u, np.ones(44), list(range(1, 43)), .1)
            for m in audit.model_ids:
                audit.record_forward(u, m, np.full(12, .5), .2)
        return audit, np.full((2, 2, 12), .5)

    def test_valid_complete_cohort(self):
        a, p = self.complete()
        a.assert_complete(p, ['u1', 'u2'], ['m1', 'm2'])
        self.assertEqual(a.summary()['forward_calls_succeeded'], 4)

    def test_missing_model_cannot_be_hidden_by_finite_fill(self):
        a, p = self.complete()
        a.records.pop(('u2', 'm2'))
        with self.assertRaises(RuntimeError):
            a.assert_complete(p, a.study_ids, a.model_ids)

    def test_failure_even_after_success_rejects(self):
        a, p = self.complete()
        a.record_failure('u2', 'decode error')
        with self.assertRaises(RuntimeError):
            a.assert_complete(p, a.study_ids, a.model_ids)

    def test_uid_order(self):
        a, p = self.complete()
        with self.assertRaises(RuntimeError):
            a.assert_complete(p, ['u2', 'u1'], a.model_ids)

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            StrictCohortAudit(['u1', 'u1'], ['m1'])
        a, _ = self.complete()
        with self.assertRaises(ValueError):
            a.record_forward('u1', 'm1', np.full(12, .5), .2)

    def test_invalid_probabilities_and_shape(self):
        for value in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                assert_prediction_vector(np.full(12, value))
        with self.assertRaises(ValueError):
            assert_prediction_vector(np.full(11, .5))
        a, p = self.complete()
        p[0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            a.assert_complete(p, a.study_ids, a.model_ids)

    def test_zero_coverage_is_not_success(self):
        a = StrictCohortAudit(['u1'], ['m1'])
        with self.assertRaises(RuntimeError):
            a.assert_complete(np.full((1, 1, 12), .5), ['u1'], ['m1'])

    def test_acquired_span_policy(self):
        self.assertEqual(cn2_full_volume_centers(np.ones(44)), list(range(1, 43)))
        edge = np.ones(44); edge[:12] = 0
        self.assertEqual(cn2_full_volume_centers(edge), list(range(13, 43)))
        internal = np.ones(44); internal[22:30] = 0
        self.assertEqual(cn2_full_volume_centers(internal), list(range(1, 43)))
        with self.assertRaises(ValueError):
            cn2_full_volume_centers(np.zeros(44))

    def test_24_windows_rejected_when_42_expected(self):
        a = StrictCohortAudit(['u1'], ['m1'])
        with self.assertRaises(ValueError):
            a.record_preparation('u1', np.ones(44), list(range(1, 25)), .1)


if __name__ == '__main__':
    unittest.main()
