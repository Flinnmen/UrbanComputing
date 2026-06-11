import unittest
import tempfile

import numpy as np

from skymonitor.agents import DroneAction, StaticAgent
from skymonitor.monitor_env import (
	MapStructure,
	TrafficData,
	TrafficMonitorEnv,
	eval_agent_repeats,
	eval_on_all_sessions,
	load_region_ids_file,
	resolve_region_ids,
)


def build_two_segment_env(
	reward_type: str,
	enters: np.ndarray = None,
	level_change_weight: np.ndarray = None,
	region_ids: np.ndarray = None,
	adjacency_matrix: np.ndarray = None,
) -> TrafficMonitorEnv:
	flow = np.array([
		[[1.0, 3.0], [3.0, 5.0]],
		[[3.0, 5.0], [1.0, 3.0]],
	])
	density = np.array([
		[[5.0, 7.0], [7.0, 9.0]],
		[[7.0, 9.0], [5.0, 7.0]],
	])
	zeros_bool = np.zeros(flow.shape, dtype=bool)
	if level_change_weight is None:
		level_change_weight = np.zeros(flow.shape, dtype=np.float32)
	traffic_data = TrafficData(
		flow=flow,
		density=density,
		segment_lengths=np.array([1.0, 1.0]),
		time_in_day=np.array([0.0, 0.5]),
		state=zeros_bool,
		score=np.zeros(flow.shape, dtype=float),
		enters=zeros_bool if enters is None else enters,
		exits=zeros_bool,
		level_state=np.ones(flow.shape, dtype=np.int8),
		level_change=level_change_weight > 0,
		level_change_weight=level_change_weight,
		region_ids=np.array([0, 1]) if region_ids is None else region_ids,
	)
	if adjacency_matrix is None:
		adjacency_matrix = np.zeros((2, 2))
	map_structure = MapStructure(
		node_coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]),
		adjacency_matrix=adjacency_matrix,
		grid_xy_of_nodes=np.array([[0, 0], [1, 0]]),
		grid_id_of_nodes=np.array([0, 1]),
		grid_xy_to_id={(0, 0): 0, (1, 0): 1},
		positions_with_data=[(0, 0), (1, 0)],
	)
	return TrafficMonitorEnv(
		data=traffic_data,
		map_structure=map_structure,
		num_drones=1,
		fixed_init_pos=[(0, 0)],
		add_historical_avg=False,
		reward_type=reward_type,
	)


class TestMonitorEnvMetrics(unittest.TestCase):
	def test_congestion_step_metrics_report_covered_change(self):
		enters = np.zeros((2, 2, 2), dtype=bool)
		enters[0, 1, :] = True
		env = build_two_segment_env('congestion', enters=enters)

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])
		metrics = info['step_metrics']

		self.assertAlmostEqual(reward, 2.0)
		self.assertAlmostEqual(metrics['reward'], 2.0)
		self.assertAlmostEqual(metrics['covered_congestion_change'], 1.0)
		self.assertAlmostEqual(metrics['total_congestion_change'], 2.0)
		self.assertIn('mfd_error', metrics)
		self.assertIn('region_mfd_error', metrics)

		episode_metrics = env.aggregate_step_metrics({key: [value] for key, value in metrics.items()})
		self.assertAlmostEqual(episode_metrics['congestion_change_coverage_rate'], 0.5)

	def test_mfd_step_metrics_report_estimation_error(self):
		env = build_two_segment_env('mfd')

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])
		metrics = info['step_metrics']

		self.assertAlmostEqual(metrics['mfd_flow_true'], 4.0)
		self.assertAlmostEqual(metrics['mfd_flow_estimate'], 3.0)
		self.assertAlmostEqual(metrics['mfd_density_true'], 8.0)
		self.assertAlmostEqual(metrics['mfd_density_estimate'], 7.0)
		self.assertAlmostEqual(metrics['mfd_error'], 0.5 * (1.0 / 4.0 + 1.0 / 8.0))
		self.assertAlmostEqual(metrics['reward'], reward)
		self.assertAlmostEqual(reward, -metrics['mfd_error'])
		self.assertIn('covered_congestion_change', metrics)
		self.assertIn('covered_congestion_level_change', metrics)

	def test_congestion_level_reward_weights_larger_jumps(self):
		small_change = np.zeros((2, 2, 2), dtype=np.float32)
		large_change = np.zeros((2, 2, 2), dtype=np.float32)
		small_change[0, 1, 0] = 1.0
		large_change[0, 1, 0] = 6.0

		env_small = build_two_segment_env('congestion_level', level_change_weight=small_change)
		env_large = build_two_segment_env('congestion_level', level_change_weight=large_change)
		env_small.reset(seed=1, options={'active_session': 0})
		env_large.reset(seed=1, options={'active_session': 0})
		_, reward_small, _, _, info_small = env_small.step([DroneAction.STAY.value])
		_, reward_large, _, _, info_large = env_large.step([DroneAction.STAY.value])

		self.assertGreater(reward_large, reward_small)
		self.assertAlmostEqual(info_small['step_metrics']['covered_congestion_level_change'], 1.0)
		self.assertAlmostEqual(info_large['step_metrics']['covered_weighted_congestion_level_change'], 6.0)


	def test_boundary_flow_reward_weights_connected_region_boundaries(self):
		adjacency = np.array([[0, 1], [1, 0]])
		env = build_two_segment_env('boundary-flow', adjacency_matrix=adjacency)

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])
		metrics = info['step_metrics']

		self.assertEqual(env.boundary_mask.tolist(), [True, True])
		self.assertAlmostEqual(reward, 6.0)
		self.assertAlmostEqual(metrics['covered_boundary_flow'], 3.0)
		self.assertAlmostEqual(metrics['total_boundary_flow'], 8.0)

		episode_metrics = env.aggregate_step_metrics({key: [value] for key, value in metrics.items()})
		self.assertAlmostEqual(episode_metrics['boundary_flow_coverage_rate'], 3.0 / 8.0)

	def test_boundary_flow_reward_keeps_plain_flow_on_non_boundary_segments(self):
		adjacency = np.array([[0, 1], [1, 0]])
		env = build_two_segment_env(
			'boundary-flow',
			region_ids=np.array([0, 0]),
			adjacency_matrix=adjacency,
		)

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])

		self.assertEqual(env.boundary_mask.tolist(), [False, False])
		self.assertAlmostEqual(reward, 3.0)
		self.assertAlmostEqual(info['step_metrics']['covered_boundary_flow'], 0.0)
		self.assertAlmostEqual(info['step_metrics']['total_boundary_flow'], 0.0)

	def test_region_mfd_reward_uses_mean_regional_error(self):
		env = build_two_segment_env('region_mfd')

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])
		metrics = info['step_metrics']

		self.assertAlmostEqual(metrics['region_0_mfd_error'], 0.0)
		self.assertAlmostEqual(metrics['region_1_mfd_error'], 1.0)
		self.assertAlmostEqual(metrics['region_mfd_error'], 0.5)
		self.assertAlmostEqual(reward, -metrics['region_mfd_error'])

	def test_balanced_reward_combines_normalized_components(self):
		enters = np.zeros((2, 2, 2), dtype=bool)
		enters[0, 1, :] = True
		level_change_weight = np.zeros((2, 2, 2), dtype=np.float32)
		level_change_weight[0, 1, 0] = 1.0
		env = build_two_segment_env(
			'balanced',
			enters=enters,
			level_change_weight=level_change_weight,
		)

		env.reset(seed=1, options={'active_session': 0})
		_, reward, _, _, info = env.step([DroneAction.STAY.value])
		metrics = info['step_metrics']

		mfd_score = 2.0 / (1.0 + metrics['mfd_error']) - 1.0
		region_mfd_score = 2.0 / (1.0 + metrics['region_mfd_error']) - 1.0
		expected_reward = (
			0.25 * 0.5
			+ 0.25 * 1.0
			+ 0.25 * mfd_score
			+ 0.25 * region_mfd_score
		)

		self.assertAlmostEqual(metrics['balanced_congestion_score'], 0.5)
		self.assertAlmostEqual(metrics['balanced_congestion_level_score'], 1.0)
		self.assertAlmostEqual(metrics['balanced_mfd_score'], mfd_score)
		self.assertAlmostEqual(metrics['balanced_region_mfd_score'], region_mfd_score)
		self.assertAlmostEqual(reward, expected_reward)

	def test_two_column_txt_region_override_uses_cluster_column(self):
		with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as fp:
			fp.write('513\t3\n512\t2\n')
			fp.flush()
			region_ids = load_region_ids_file(fp.name)

		class Dataset:
			section_ids_sorted = np.array([512, 513])

		self.assertEqual(resolve_region_ids(region_ids, Dataset()).tolist(), [2, 3])

	def test_eval_repeats_return_metric_arrays(self):
		enters = np.zeros((2, 2, 2), dtype=bool)
		enters[0, 1, :] = True
		env = build_two_segment_env('congestion', enters=enters)
		agent = StaticAgent(num_drones=1)

		eval_res = eval_on_all_sessions(env, agent, seed=1)
		self.assertEqual(eval_res['all_reward'], [2.0, 0.0])
		self.assertEqual(eval_res['all_metrics']['covered_congestion_change'], [1.0, 0.0])
		self.assertEqual(eval_res['all_metrics']['total_congestion_change'], [2.0, 0.0])
		self.assertEqual(eval_res['all_metrics']['congestion_change_coverage_rate'], [0.5, 0.0])

		stats, reward_reps, metric_reps, _ = eval_agent_repeats(env, agent, eval_repeat=1, eval_seed=1)
		self.assertEqual(reward_reps.tolist(), [[2.0, 0.0]])
		self.assertEqual(metric_reps['covered_congestion_change'].tolist(), [[1.0, 0.0]])
		self.assertIn('mfd_error', metric_reps)
		self.assertIn('region_mfd_error', metric_reps)
		self.assertIn('covered_congestion_level_change', metric_reps)
		self.assertEqual(stats['avg_covered_congestion_change_per_session'], [1.0, 0.0])


if __name__ == '__main__':
	unittest.main()
