import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from bayes_opt import BayesianOptimization

from skymonitor.agents import PolicyAgent
from skymonitor.monitor_env import build_traffic_monitor_env, eval_agent_repeats, load_region_ids_file
from skymonitor.rl_drone import POLICY_NETS, train_monitoring_agent_with_ppo
from skymonitor.simbarca_explore import initialize_dataset


def _json_ready(value):
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		return {key: _json_ready(item) for key, item in value.items()}
	if isinstance(value, list):
		return [_json_ready(item) for item in value]
	return value


def optimize_congestion_weights(args) -> dict:
	log_path = Path(args.logdir)
	log_path.mkdir(parents=True, exist_ok=True)

	region_ids = load_region_ids_file(args.region_ids_file) if args.region_ids_file is not None else None
	trainset, testset = initialize_dataset()
	history = []

	def objective(w12: float, w23: float, w34: float) -> float:
		weights = (float(w12), float(w23), float(w34))
		trial_id = len(history)
		trial_dir = log_path / f'trial_{trial_id:03d}'
		model = train_monitoring_agent_with_ppo(
			policy_type=args.policy_type,
			total_timesteps=args.train_timesteps,
			num_envs=args.num_envs,
			num_drones=args.num_drones,
			learning_rate=args.learning_rate,
			log_dir=str(trial_dir),
			save_name=args.save_name,
			seed=args.train_seed,
			eval_freq=0,
			eval_repeat=args.eval_repeat,
			eval_seed=args.eval_seed,
			norm_obs=args.norm_obs,
			reward_type='congestion_level',
			congestion_level_thresholds=tuple(args.congestion_level_thresholds),
			level_change_weights=weights,
			region_ids=region_ids,
		)
		env = build_traffic_monitor_env(
			trainset=trainset,
			testset=testset,
			num_drones=args.num_drones,
			env_type='test',
			norm_obs=args.norm_obs,
			reward_type='congestion_level',
			congestion_level_thresholds=tuple(args.congestion_level_thresholds),
			level_change_weights=weights,
			region_ids=region_ids,
		)
		with torch.no_grad():
			stats, _, _, _ = eval_agent_repeats(
				env=env,
				agent=PolicyAgent(policy=model),
				eval_repeat=args.eval_repeat,
				eval_seed=args.eval_seed,
			)
		env.close()

		mfd_error = float(stats['avg_mfd_error'])
		target = -mfd_error
		history.append({
			'trial': trial_id,
			'weights': {'w12': weights[0], 'w23': weights[1], 'w34': weights[2]},
			'target': target,
			'avg_mfd_error': mfd_error,
			'stats': stats,
		})
		with (log_path / 'optimization_results.json').open('w') as fp:
			json.dump({'history': _json_ready(history)}, fp, indent=4)
		return target

	bounds = tuple(args.weight_bounds)
	optimizer = BayesianOptimization(
		f=objective,
		pbounds={'w12': bounds, 'w23': bounds, 'w34': bounds},
		random_state=args.bo_seed,
	)
	optimizer.maximize(init_points=args.init_points, n_iter=args.n_iter)

	results = {
		'best': _json_ready(optimizer.max),
		'history': _json_ready(history),
		'optimizer_results': _json_ready(optimizer.res),
	}
	with (log_path / 'optimization_results.json').open('w') as fp:
		json.dump(results, fp, indent=4)
	return results


def parse_args():
	parser = argparse.ArgumentParser(
		description='Optimize multi-level congestion reward weights with Bayesian Optimization.',
	)
	parser.add_argument('--policy-type', type=str, default='simple', choices=sorted(POLICY_NETS))
	parser.add_argument('--train-timesteps', type=int, default=100000)
	parser.add_argument('--num-envs', type=int, default=8)
	parser.add_argument('--num-drones', type=int, default=10)
	parser.add_argument('--learning-rate', type=float, default=1e-4)
	parser.add_argument('--logdir', type=str, default='scratch/drone_monitor_bo')
	parser.add_argument('--save-name', type=str, default='model')
	parser.add_argument('--train-seed', type=int, default=42)
	parser.add_argument('--eval-repeat', type=int, default=5)
	parser.add_argument('--eval-seed', type=int, default=888)
	parser.add_argument('--bo-seed', type=int, default=0)
	parser.add_argument('--init-points', type=int, default=3)
	parser.add_argument('--n-iter', type=int, default=10)
	parser.add_argument('--weight-bounds', type=float, nargs=2, default=(0.0, 5.0))
	parser.add_argument('--norm-obs', action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument('--congestion-level-thresholds', type=float, nargs=3, default=(0.25, 0.5, 0.75))
	parser.add_argument('--region-ids-file', type=str, default=None)
	parser.add_argument(
		'--log-level',
		type=str,
		default='info',
		choices=['debug', 'info', 'warning', 'error', 'critical'],
	)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	logging.basicConfig(level=getattr(logging, args.log_level.upper()))
	logging.info('Arguments: %s', args)
	results = optimize_congestion_weights(args)
	logging.info('Best congestion level weights: %s', results['best'])
