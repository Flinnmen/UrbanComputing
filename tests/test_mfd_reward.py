import unittest

import numpy as np

from skymonitor.mfd_reward import (
	calculate_mfd_metrics,
	calculate_mfd_reward,
	calculate_region_mfd_metrics,
	estimate_mfd,
)


class TestMFDReward(unittest.TestCase):
	def test_all_segments_covered_has_zero_error(self):
		flow = np.array([1.0, 2.0])
		density = np.array([3.0, 4.0])
		lengths = np.array([10.0, 30.0])
		coverage = np.array([1, 1])

		self.assertAlmostEqual(calculate_mfd_reward(flow, density, lengths, coverage), 0.0)

	def test_partial_coverage_uses_length_weighting(self):
		flow = np.array([1.0, 10.0, 100.0])
		density = np.array([2.0, 20.0, 200.0])
		lengths = np.array([1.0, 9.0, 90.0])
		coverage = np.array([1, 1, 0])

		est_dist, est_time = estimate_mfd(flow, density, lengths, coverage)
		self.assertAlmostEqual(est_dist, 9.1)
		self.assertAlmostEqual(est_time, 18.2)

		true_dist, true_time = estimate_mfd(flow, density, lengths)
		expected_reward = -0.5 * (
			abs(est_dist - true_dist) / abs(true_dist)
			+ abs(est_time - true_time) / abs(true_time)
		)
		self.assertAlmostEqual(calculate_mfd_reward(flow, density, lengths, coverage), expected_reward)

		metrics = calculate_mfd_metrics(flow, density, lengths, coverage)
		self.assertAlmostEqual(metrics['mfd_flow_true'], true_dist)
		self.assertAlmostEqual(metrics['mfd_flow_estimate'], est_dist)
		self.assertAlmostEqual(metrics['mfd_density_true'], true_time)
		self.assertAlmostEqual(metrics['mfd_density_estimate'], est_time)
		self.assertAlmostEqual(metrics['mfd_error'], -expected_reward)
		self.assertAlmostEqual(metrics['reward'], expected_reward)

	def test_zero_coverage_uses_zero_estimate(self):
		flow = np.array([1.0, 3.0])
		density = np.array([2.0, 4.0])
		lengths = np.array([1.0, 1.0])
		coverage = np.array([0, 0])

		self.assertEqual(estimate_mfd(flow, density, lengths, coverage), (0.0, 0.0))
		self.assertAlmostEqual(calculate_mfd_reward(flow, density, lengths, coverage), -1.0)

	def test_region_mfd_metrics_average_region_errors(self):
		flow = np.array([1.0, 10.0])
		density = np.array([2.0, 20.0])
		lengths = np.array([1.0, 1.0])
		coverage = np.array([1, 0])
		region_ids = np.array([0, 1])

		metrics = calculate_region_mfd_metrics(flow, density, lengths, coverage, region_ids)

		self.assertAlmostEqual(metrics['region_0_mfd_error'], 0.0)
		self.assertAlmostEqual(metrics['region_1_mfd_error'], 1.0)
		self.assertAlmostEqual(metrics['region_mfd_error'], 0.5)
		self.assertAlmostEqual(metrics['region_mfd_reward'], -0.5)


if __name__ == '__main__':
	unittest.main()
