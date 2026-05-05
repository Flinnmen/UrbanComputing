import unittest

import numpy as np

from skymonitor.agents import DroneAction, StaticAgent
from skymonitor.monitor_env import (
	MapStructure,
	TrafficData,
	TrafficMonitorEnv,
	eval_agent_repeats,
	eval_on_all_sessions,
)


def build_two_segment_env(reward_type: str, enters: np.ndarray = None) -> TrafficMonitorEnv:
	flow = np.array([
		[[1.0, 3.0], [3.0, 5.0]],
		[[3.0, 5.0], [1.0, 3.0]],
	])
	density = np.array([
		[[5.0, 7.0], [7.0, 9.0]],
		[[7.0, 9.0], [5.0, 7.0]],
	])
	zeros_bool = np.zeros(flow.shape, dtype=bool)
	traffic_data = TrafficData(
		flow=flow,
		density=density,
		segment_lengths=np.array([1.0, 1.0]),
		time_in_day=np.array([0.0, 0.5]),
		state=zeros_bool,
		score=np.zeros(flow.shape, dtype=float),
		enters=zeros_bool if enters is None else enters,
		exits=zeros_bool,
	)
	map_structure = MapStructure(
		node_coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]),
		adjacency_matrix=np.zeros((2, 2)),
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
		self.assertEqual(stats['avg_covered_congestion_change_per_session'], [1.0, 0.0])


if __name__ == '__main__':
	unittest.main()
