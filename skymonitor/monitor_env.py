"""Gymnasium environment for SkyMonitor.
see the `Gymnasium environment creation tutorial` at the link:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation
"""

import argparse
import json
import logging
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch

from skymonitor.simbarca_explore import SimBarcaExplore, initialize_dataset
from skymonitor.agents import (
	BaseAgent,
	RandomAgent,
	StaticAgent,
	SweepingAgent,
	DroneAction,
	ActionToDirection,
)
from skymonitor.congestion import get_congestion_score, get_congestion_change
from skymonitor.mfd_reward import calculate_mfd_metrics
from skytraffic.utils.structure import build_grid_xy_to_id

logger = logging.getLogger(__name__)

# make them un-mutable dataclasses for safety
@dataclass(frozen=True)
class TrafficData:
	flow: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)
	density: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)
	segment_lengths: np.ndarray = None # shape (num_positions, )
	time_in_day: np.ndarray = None # shape (num_time_steps, )
	state: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)
	score: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)
	enters: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)
	exits: np.ndarray = None # shape (num_sessions, num_time_steps, num_positions)

# make them un-mutable dataclasses for safety
@dataclass(frozen=True)
class MapStructure:
	# the (x, y) coordinates of each node, shape (num_nodes, 2)
	node_coordinates: np.ndarray = None
	# the adjacency matrix of the grid, shape (num_nodes, num_nodes)
	adjacency_matrix: np.ndarray = None
	# cell size in meters
	cell_size: float = 220.0
	# the (x, y) coordinates of each node in the grid, shape (num_nodes, 2)
	grid_xy_of_nodes: np.ndarray = None
	# the grid ID for each node, shape (num_nodes, )
	grid_id_of_nodes: np.ndarray = None
	grid_xy_to_id: Dict[Tuple[int, int], int] = None
	positions_with_data: List[Tuple[int, int]] = None

	def __post_init__(self):
		if self.grid_xy_to_id is None:
			assert self.grid_xy_of_nodes is not None and self.grid_id_of_nodes is not None, "grid_xy and grid_id must be provided"
			grid_xy_to_id = build_grid_xy_to_id(self.grid_xy_of_nodes, self.grid_id_of_nodes)
			object.__setattr__(self, 'grid_xy_to_id', grid_xy_to_id)
		if self.positions_with_data is None:
			positions = [] if self.grid_xy_to_id is None else list(self.grid_xy_to_id.keys())
			object.__setattr__(self, 'positions_with_data', positions)

class ObservationNormalizer:

	def __init__(self, traffic_data: TrafficData, norm_by_pos:bool = False):
		self.flow_mean: np.ndarray = None
		self.flow_std: np.ndarray = None
		self.density_mean: np.ndarray = None
		self.density_std: np.ndarray = None
		self.time_mean: np.ndarray = None
		self.time_std: np.ndarray = None

		self.fit(norm_by_pos, traffic_data)

	def fit(self, norm_by_pos: bool, traffic_data: TrafficData):
		axis = (0, 1) if norm_by_pos else None
		self.flow_mean = traffic_data.flow.mean(axis=axis)
		self.flow_std = traffic_data.flow.std(axis=axis) + 1e-6
		self.density_mean = traffic_data.density.mean(axis=axis)
		self.density_std = traffic_data.density.std(axis=axis) + 1e-6
		self.time_mean = traffic_data.time_in_day.mean()
		self.time_std = traffic_data.time_in_day.std() + 1e-6

	def normalize(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

		norm_obs = observation.copy()

		if self.flow_mean is not None and self.flow_std is not None:
			norm_obs['flow'] = (observation['flow'] - self.flow_mean) / self.flow_std
			if 'ha_flow' in observation:
				norm_obs['ha_flow'] = (observation['ha_flow'] - self.flow_mean) / self.flow_std
		if self.density_mean is not None and self.density_std is not None:
			norm_obs['density'] = (observation['density'] - self.density_mean) / self.density_std
			if 'ha_density' in observation:
				norm_obs['ha_density'] = (observation['ha_density'] - self.density_mean) / self.density_std
		if self.time_mean is not None and self.time_std is not None:
			norm_obs['time_in_day'] = (observation['time_in_day'] - self.time_mean) / self.time_std

		return norm_obs
	
	def scale_back(self, norm_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

		orig_obs = norm_obs.copy()

		if self.flow_mean is not None and self.flow_std is not None:
			orig_obs['flow'] = norm_obs['flow'] * self.flow_std + self.flow_mean
			if 'ha_flow' in norm_obs:
				orig_obs['ha_flow'] = norm_obs['ha_flow'] * self.flow_std + self.flow_mean
		if self.density_mean is not None and self.density_std is not None:
			orig_obs['density'] = norm_obs['density'] * self.density_std + self.density_mean
			if 'ha_density' in norm_obs:
				orig_obs['ha_density'] = norm_obs['ha_density'] * self.density_std + self.density_mean
		if self.time_mean is not None and self.time_std is not None:
			orig_obs['time_in_day'] = norm_obs['time_in_day'] * self.time_std + self.time_mean

		return orig_obs

class TrafficMonitorEnv(gym.Env):
	"""
	Design principles for the environment (following Gymnasium API):
	- **🎯 Agent Skill**: Collect traffic data in a grid map
	- **👀 Information**: Observed traffic states, agent position, map structure
	- **🎮 Actions**: Move up, down, left or right by 1 or 2 steps; Move diagonally by 1 step; Don't move
	- **🏆 Success**: maximize the defined reward
	- **⏰ End**: A simulation session ends.

	Init Args:
		data: TrafficData, the traffic data container
		map_structure: MapStructure, the map structure container
		num_drones: int, number of drones in the environment
		seed: int, random seed for environment initialization
		fixed_init_pos: List[Tuple[int, int]], fixed initial positions for drones
		normalizer: ObservationNormalizer, the normalizer for observation values (flow, density and timestamp)
	"""
	patched_pos = [(7, 2), (8, 3), (9, 3), (9, 4), (17, 11)]
	metadata = {'render_modes': ['human'], 'name': 'drone_traffic_v0', 'render_fps': 30}

	def __init__(
		self,
		data: TrafficData,
		map_structure: MapStructure,
		num_drones: int,
		fixed_init_pos: List[Tuple[int, int]] = None,
		obs_normalizer: ObservationNormalizer = None,
		add_historical_avg: bool = True,
		reward_type: str = 'congestion',
		seed: int = None,
	):
		super().__init__()
		# data container
		self.traffic_data = data
		# map structure info
		self.map_structure = map_structure
		# settings and configurations
		self.num_drones = int(num_drones)
		# the random seed to be used upon next reset, in fact, each active session can be considered as a separate environment
		# we want each reset to use a different active session and initial locations, so the agent can explore this dataset better
		self.next_reset_seed: int = seed
		# always init the drones at fixed set of init_positions if provided
		self.fixed_init_pos = fixed_init_pos
		self.obs_normalizer = obs_normalizer
		self.add_historical_avg = add_historical_avg
		self.reward_type = reward_type

		# spaces of the environment
		self.action_space = spaces.MultiDiscrete([len(DroneAction)] * num_drones)
		S, T, P = self.traffic_data.flow.shape
		if self.obs_normalizer is not None:
			z_low, z_high = -10.0, 10.0 # reasonable range after normalization
			flow_space = spaces.Box(low=z_low, high=z_high, shape=(P, ), dtype=np.float32)
			density_space = spaces.Box(low=z_low, high=z_high, shape=(P, ), dtype=np.float32)
			time_space = spaces.Box(low=z_low, high=z_high, shape=(1, ), dtype=np.float32)
		else:
			flow_space = spaces.Box(low=0.0, high=2.0, shape=(P, ), dtype=np.float32)
			density_space = spaces.Box(low=0.0, high=2.0, shape=(P, ), dtype=np.float32)
			time_space = spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
		obs_space_dict = {
				# observed traffic data and time info
				'flow': flow_space,
				'density': density_space,
				'time_in_day': time_space,
				# the coordinates of drones and the node coverage mask
				'positions_x': spaces.MultiDiscrete(
					[int(self.map_structure.grid_xy_of_nodes[:, 0].max()) + 1] * self.num_drones
				),
				'positions_y': spaces.MultiDiscrete(
					[int(self.map_structure.grid_xy_of_nodes[:, 1].max()) + 1] * self.num_drones
				),
				'coverage_mask': spaces.MultiBinary(n=P),
			}
		if self.add_historical_avg:
			obs_space_dict.update({
				'ha_flow': flow_space,
				'ha_flow_std': spaces.Box(low=0.0, high=1.0, shape=(P, ), dtype=np.float32),
				'ha_density': density_space,
				'ha_density_std': spaces.Box(low=0.0, high=1.0, shape=(P, ), dtype=np.float32),
			})
		self.observation_space = spaces.Dict(obs_space_dict)

		self.total_sessions = S
		self.total_steps = T
		self.total_locations = P

		# running states
		self.step_index: int = 0
		self.data_sample: Dict[str, torch.Tensor] = None
		self.positions: List[Tuple[int, int]] = []
		self.positions_history = list()
		self.observation_history: List[Dict[str, torch.Tensor]] = list()
		self._active_session = None

		# pre-compute some useful statistics
		self.avg_flow_by_location = self.traffic_data.flow.mean(axis=(0,1))  # shape (P)
		# historical average flow and density (and their stds), useful when no observation is available
		self.ha_flow = self.traffic_data.flow.mean(axis=0)
		self.ha_flow_std = self.traffic_data.flow.std(axis=0)
		self.ha_flow_std = self.ha_flow_std / self.ha_flow_std.max() # normalize to (0, 1)
		self.ha_density = self.traffic_data.density.mean(axis=0)
		self.ha_density_std = self.traffic_data.density.std(axis=0)
		self.ha_density_std = self.ha_density_std / self.ha_density_std.max() # normalize to (0, 1)

		# amend map structure to make the shape smoother
		self.all_positions = sorted(list(set(self.map_structure.positions_with_data).union(set(self.patched_pos))))

	def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
		"""Resets environment and returns observations and infos for all agents."""
		self.clear_state()
		# each time we reset the environment, we use a new random seed because many things, including
		# the drone locations, are generated using this seed, and we want the agent to explore
		# different initial positions rather than starting always from the same ones.
		super().reset(seed=seed if seed is not None else self.next_reset_seed)
		self.next_reset_seed = self.np_random.integers(100_000_000, 999_999_999).item()

		# accept override from options if provided, otherwise randomly select active session(s)
		active_session = None if not isinstance(options, dict) else options.get('active_session', None)
		if active_session is None:
			active_session = self.np_random.integers(0, self.total_sessions).item()
		self.active_session = active_session
		logger.debug(f'Environment reset with seed: {self.np_random_seed} and active session: {self.active_session}.')

		self.data_iterator = self.session_data_iterator()
		self.step_index, self.data_sample = next(self.data_iterator)
		assert self.step_index == 0, 'Step index should start from 0 after reset.'

		positions = self.init_agent_positions()
		obs = self.get_observations(positions)

		self.update_history(positions, obs)
		info = self.get_info()

		return obs, info


	def step(self, actions: List[DroneAction]) -> Tuple[Dict, float, bool, bool, Dict]:
		"""In the previous step, the agent took `actions` to move the drones to new positions.
		The reward is computed from the new observation.
		"""
		try:
			self.step_index, self.data_sample = next(self.data_iterator)
		# this is never reached normally since we check the termination flag with total_steps
		except StopIteration:
			return dict(), 0.0, True, False, {}

		pos = self.apply_actions(actions)
		obs = self.get_observations(pos)

		# the reward is calculated based on the current observation
		step_metrics = self.calculate_step_metrics(obs)
		reward_value = step_metrics['reward']

		terminated_flag = self.step_index >= (self.total_steps - 1)
		truncated_flag = False

		self.update_history(pos, obs)
		info = self.get_info()
		info['step_metrics'] = step_metrics

		return obs, reward_value, terminated_flag, truncated_flag, info

	def close(self):
		self.clear_state()

	def clear_state(self) -> None:
		self.step_index = 0
		self.data_sample = None
		self.positions.clear()
		self.positions_history.clear()
		self.observation_history.clear()

	@property
	def active_session(self):
		return self._active_session

	@active_session.setter
	def active_session(self, idx):
		try:
			idx = int(idx)
		except Exception as e:
			raise ValueError(f"Active session should be an integer: {e}")
		self._active_session = idx % self.total_sessions

	def session_data_iterator(self):
		for i in range(self.total_steps):
			sample = {
				'flow': self.traffic_data.flow[self.active_session, i, :],
				'density': self.traffic_data.density[self.active_session, i, :],
				# ensure the shape is (1, ), match the observation space definition
				'time_in_day': np.atleast_1d(self.traffic_data.time_in_day[i]).astype(np.float32),
			}
			yield i, sample

	def calculate_reward(self, observation: Dict[str, np.ndarray]) -> float:
		"""Define the reward based on the current observation."""
		return self.calculate_step_metrics(observation)['reward']

	def calculate_step_metrics(self, observation: Dict[str, np.ndarray]) -> Dict[str, float]:
		"""Compute reward and reportable physical metrics for the current step."""
		coverage = observation['coverage_mask'].astype(np.float32)
		if self.reward_type == 'mfd':
			return calculate_mfd_metrics(
				flow=self.traffic_data.flow[self.active_session, self.step_index, :],
				density=self.traffic_data.density[self.active_session, self.step_index, :],
				segment_lengths=self.traffic_data.segment_lengths,
				coverage_mask=coverage,
			)

		enters = self.traffic_data.enters[self.active_session, self.step_index, :].astype(np.float32)
		exits = self.traffic_data.exits[self.active_session, self.step_index, :].astype(np.float32)
		congestion_change = enters + exits
		covered_congestion_change = (coverage * congestion_change).sum()
		total_congestion_change = congestion_change.sum()
		reward = (self.avg_flow_by_location * coverage * congestion_change).sum()

		return {
			'covered_congestion_change': float(covered_congestion_change.item()),
			'total_congestion_change': float(total_congestion_change.item()),
			'reward': float(reward.item()),
		}

	def get_step_metric_reducers(self) -> Dict[str, str]:
		if self.reward_type == 'mfd':
			return {
				'mfd_flow_true': 'mean',
				'mfd_flow_estimate': 'mean',
				'mfd_flow_abs_error': 'mean',
				'mfd_flow_rel_error': 'mean',
				'mfd_density_true': 'mean',
				'mfd_density_estimate': 'mean',
				'mfd_density_abs_error': 'mean',
				'mfd_density_rel_error': 'mean',
				'mfd_error': 'mean',
				'reward': 'sum',
			}

		return {
			'covered_congestion_change': 'sum',
			'total_congestion_change': 'sum',
			'reward': 'sum',
		}

	def aggregate_step_metrics(self, step_metrics: Dict[str, List[float]]) -> Dict[str, float]:
		reducers = self.get_step_metric_reducers()
		episode_metrics = {}
		for key, values in step_metrics.items():
			values = np.asarray(values, dtype=np.float64)
			if reducers[key] == 'sum':
				episode_metrics[key] = float(values.sum())
			elif reducers[key] == 'mean':
				episode_metrics[key] = float(values.mean())
			else:
				raise ValueError(f"Invalid reducer {reducers[key]} for metric {key}.")

		if 'covered_congestion_change' in episode_metrics and 'total_congestion_change' in episode_metrics:
			total_change = episode_metrics['total_congestion_change']
			episode_metrics['congestion_change_coverage_rate'] = (
				episode_metrics['covered_congestion_change'] / total_change if total_change > 0 else 0.0
			)

		return episode_metrics

	def init_agent_positions(self):
		"""Initialize drone positions at start of episode."""
		if self.fixed_init_pos is not None:
			assert len(self.fixed_init_pos) == self.num_drones, "Length of fixed_init_pos must match num_drones"
			# MUST BE a deep copy otherwise the positions will be modified in place during the episode
			return deepcopy(self.fixed_init_pos)

		interior_positions = [
			(x, y) for x, y in self.map_structure.positions_with_data
			if (x + 1, y) in self.map_structure.positions_with_data
			and (x - 1, y) in self.map_structure.positions_with_data
			and (x, y + 1) in self.map_structure.positions_with_data
			and (x, y - 1) in self.map_structure.positions_with_data
		]
		indices = self.np_random.choice(len(interior_positions), size=self.num_drones, replace=False)
		return [interior_positions[idx] for idx in indices]

	def compute_coverage_mask(self, positions: List[Tuple[int, int]]) -> np.ndarray:
		mask = sum(
			[
				self.map_structure.grid_id_of_nodes == self.map_structure.grid_xy_to_id.get((x, y), -1)
				for x, y in positions
			]
		)
		return (mask > 0).astype(np.int8)

	def get_observations(self, positions: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
		"""get the observed traffic at the given positions."""

		# clean data from the dataset, for 1 time step
		# make a deep copy to avoid modifying the original data sample
		observed_traffic = deepcopy(self.data_sample)
		coverage_mask = self.compute_coverage_mask(positions)
		positions_x = np.array([pos[0] for pos in positions], dtype=np.int64)
		positions_y = np.array([pos[1] for pos in positions], dtype=np.int64)

		# now we modify the nan values to be 0 to align with the observation space definition
		for key, value in observed_traffic.items():
			if not isinstance(value, np.ndarray):
				continue
			if value.shape != coverage_mask.shape:
				continue
			observed_traffic[key] = np.nan_to_num(value, nan=0.0, copy=False)
			observed_traffic[key][coverage_mask == 0] = 0.0

		observation = {
			**observed_traffic,
			'coverage_mask': coverage_mask,
			'positions_x': positions_x,
			'positions_y': positions_y,
		}

		if self.add_historical_avg:
			observation.update(self.get_historical_average())

		if self.obs_normalizer is not None:
			observation = self.obs_normalizer.normalize(observation)

		return observation

	def get_historical_average(self) -> Dict[str, np.ndarray]:
		time_step = self.step_index
		return {
			'ha_flow': self.ha_flow[time_step, :],
			'ha_flow_std': self.ha_flow_std[time_step, :],
			'ha_density': self.ha_density[time_step, :],
			'ha_density_std': self.ha_density_std[time_step, :],
		}
	
	def get_info(self) -> Dict:
		return {
			'step_index': self.step_index,
			'session_index': self.active_session,
			'positions': list(self.positions),
			'state': self.traffic_data.state[self.active_session, self.step_index, :],
			'congestion_score': self.traffic_data.score[self.active_session, self.step_index, :],
			'enters': self.traffic_data.enters[self.active_session, self.step_index, :],
			'exits': self.traffic_data.exits[self.active_session, self.step_index, :],
		}

	def update_history(self, positions: List[Tuple[int, int]], observation: Dict[str, np.ndarray]) -> None:
		self.positions = positions
		self.positions_history.append(tuple(self.positions))
		self.observation_history.append(observation)

		assert len(self.observation_history) == (self.step_index + 1), 'Observation history length mismatch.'

	def apply_actions(self, actions: np.array) -> List[Tuple[int, int]]:

		def _to_act_space(arr: np.ndarray) -> List[DroneAction]:
			return [DroneAction(a) for a in np.asarray(arr).tolist()]
		
		new_positions: List[Tuple[int, int]] = []
		occupied = set(self.positions) # avoid drone overlapping

		for idx, act in enumerate(_to_act_space(actions)):
			old_pos = self.positions[idx]
			new_pos = self._get_new_position(act, old_pos)
			# stay if the new position is already occupied
			if new_pos in occupied:  
				new_pos = old_pos
			# move to new position
			else:
				new_pos = new_pos
				occupied.add(new_pos)
				occupied.remove(old_pos)  # free up the old position
			new_positions.append(new_pos)

		assert len(new_positions) == len(set(new_positions)), "Two drones cannot occupy the same position."

		return new_positions


	def _get_new_position(self, action: DroneAction, old_pos: Tuple[int, int]) -> Tuple[int, int]:
		"""works for 1 single position update"""
		direction = ActionToDirection.get(action, (0, 0))
		new_pos = (old_pos[0] + direction[0], old_pos[1] + direction[1])
		if new_pos not in self.all_positions:
			new_pos = old_pos
		return new_pos

	def render(self):
		pass

def prepare_env_data_structure(
	dataset: SimBarcaExplore,
	density_threshold=0.001,
	ff_quantile=0.85,
	entry_thd=0.5,
	exit_thd=0.7,
):
	# the time window and time step setting are not relevant to RL environment
	# as they are related to generating indexes for data sampling batch-wise training
	# in RL env, we take the data sequences directly and move 1 step each time, so no need for the complication
	# norm_tid=True normalizes the time in day encoding [8am, 10am] to have zero mean and unit variance
	flow, density = dataset.veh_flow_3min, dataset.veh_density_3min

	congestion_score = get_congestion_score(
		flow=flow,
		density=density,
		density_threshold=density_threshold,
		ff_quantile=ff_quantile,
	)
	state, enters, exits = get_congestion_change(congestion_score, entry_thd=entry_thd, exit_thd=exit_thd)

	traffic_data = TrafficData(
		flow=flow,
		density=density,
		segment_lengths=dataset.segment_lengths,
		time_in_day=dataset.time_in_day_3min,
		state=state,
		score=congestion_score,
		enters=enters,
		exits=exits,
	)
	map_structure = MapStructure(
		node_coordinates=dataset.node_coordinates,
		adjacency_matrix=dataset.adjacency,
		grid_xy_of_nodes=dataset.grid_xy,
		grid_id_of_nodes=dataset.grid_id,
		grid_xy_to_id=dataset.grid_xy_to_id,
	)

	return traffic_data, map_structure


def build_traffic_monitor_env(
	trainset: SimBarcaExplore = None,
	testset: SimBarcaExplore = None,
	num_drones: int = 5,
	env_type: str = 'train',
	norm_obs: bool = True,
	reward_type: str = 'congestion',
) -> Tuple[TrafficMonitorEnv, Optional[TrafficMonitorEnv]]:
	"""
	Build the traffic monitoring envs.

	Note that the test set needs to be build using the normalizer computed from the train env

	Args:
		num_drones (int): number of drones
		env_type (str, optional): type of environment to build. One of 'train', 'test', or 'both'. Defaults to 'train'.
		norm_obs (bool, optional): whether to normalize observations. Defaults to True.
		reward_type (str, optional): reward definition. One of 'congestion' or 'mfd'. Defaults to 'congestion'.
	Returns:
		Tuple[TrafficMonitorEnv, Optional[TrafficMonitorEnv]]: train and test environments
	"""
	if trainset is None:
		raise ValueError("trainset must always be provided (for normalizer computation of test environments).")
	if env_type in ['test', 'both'] and testset is None:
		raise ValueError("testset must be provided when env_type is 'test' or 'both'.")

	# by default build the training environment
	train_traffic_data, train_map_structure = prepare_env_data_structure(trainset)
	normalizer = ObservationNormalizer(train_traffic_data)

	train_env = TrafficMonitorEnv(
		data=train_traffic_data,
		map_structure=train_map_structure,
		obs_normalizer=normalizer if norm_obs else None,
		num_drones=num_drones,
		reward_type=reward_type,
	)

	# if requested, build the test environment
	if env_type in ['test', 'both']:
		test_traffic_data, test_map_structure = prepare_env_data_structure(testset)
		test_env = TrafficMonitorEnv(
			data=test_traffic_data,
			map_structure=test_map_structure,
			obs_normalizer=normalizer if norm_obs else None,
			num_drones=num_drones,
			reward_type=reward_type,
		)

	if env_type == 'train':
		return train_env
	elif env_type == 'test':
		return test_env
	elif env_type == 'both':
		return train_env, test_env
	else:
		raise ValueError(f"Invalid envs argument: {env_type}. Must be 'train', 'test' or 'both'.")

@dataclass(frozen=True)
class _EvalInitState:
	positions: List[Tuple[int, int]]
	sweeping_progress: List[int]


def _build_sweeping_eval_init_state(
	agent: SweepingAgent,
	rng: np.random.Generator,
) -> _EvalInitState:
	positions = []
	init_progress = []
	occupied = set()
	for agent_id, trajectory in enumerate(agent.trajectory_plan):
		if not trajectory:
			raise ValueError(f"No available initial position on sweeping route for agent {agent_id}.")
	common_progress = int(rng.integers(min(len(trajectory) for trajectory in agent.trajectory_plan)))
	for agent_id, trajectory in enumerate(agent.trajectory_plan):
		candidates = sorted(
			[idx for idx, pos in enumerate(trajectory) if pos not in occupied],
			key=lambda idx: abs(idx - common_progress),
		)
		if not candidates:
			raise ValueError(f"No available initial position on sweeping route for agent {agent_id}.")
		progress = candidates[0]
		pos = trajectory[progress]
		positions.append(pos)
		init_progress.append(progress)
		occupied.add(pos)
	return _EvalInitState(positions=positions, sweeping_progress=init_progress)


def eval_on_all_sessions(
	env: TrafficMonitorEnv,
	agent: BaseAgent,
	seed: int = 42,
	init_state: _EvalInitState = None,
) -> Dict[str, List]:
	# seed the environment once before the loop to initialize its random number generator.
	# in this way the drones are randomly spawn for each session, but remains consistent across runs
	env.reset(seed=seed)  
	eval_res = {
		'all_reward': [],
		'all_metrics': defaultdict(list),
		'all_trajectories': [],
	}

	for active_session in range(env.total_sessions):
		# print('=== Running agents on session {} ==='.format(active_session))
		observation, info = env.reset(options={'active_session': active_session})
		if init_state is not None:
			logger.info(f"Overriding the positions to: {init_state.positions}")
			env.positions_history.clear()
			env.observation_history.clear()
			env.positions = init_state.positions # move the drones to high-reward positions
			observation = env.get_observations(env.positions)
			env.update_history(env.positions, observation)
			info = env.get_info()
		# this is bad in general because the agents should be Markov (e.g., doesn't depend on past history)
		if callable(getattr(agent, 'clear_state', None)):
			agent.clear_state()
		if init_state is not None:
			agent.trajectory_progress = list(init_state.sweeping_progress)

		done = False
		truncated = False
		episode_reward = 0.0
		episode_step_metrics = defaultdict(list)
		while not (done or truncated):
			actions = agent.select_action(observation)
			observation, reward, done, truncated, info = env.step(actions)
			episode_reward += reward
			for key, value in info['step_metrics'].items():
				episode_step_metrics[key].append(float(value))

		eval_res['all_reward'].append(episode_reward)
		episode_metrics = env.aggregate_step_metrics(episode_step_metrics)
		for key, value in episode_metrics.items():
			if key != 'reward':
				eval_res['all_metrics'][key].append(value)
		eval_res['all_trajectories'].append([pos for pos in env.positions_history])

	return eval_res


def build_trivial_monitoring_agent(
	agent_type: str,
	num_drones: int,
	sweeping_pos: List[Tuple[int, int]] = None,
) -> BaseAgent:
	if agent_type == 'random':
		return RandomAgent(num_drones=num_drones)
	if agent_type == 'static':
		return StaticAgent(num_drones=num_drones)
	if agent_type == 'sweeping':
		if sweeping_pos is None:
			raise ValueError("sweeping_pos must be provided for sweeping agents.")
		return SweepingAgent(num_drones=num_drones, sweeping_pos=sweeping_pos)
	raise ValueError(f"Invalid trivial agent type: {agent_type}. Must be 'random', 'static' or 'sweeping'.")


def eval_agent_repeats(
	env: TrafficMonitorEnv,
	agent: BaseAgent,
	eval_repeat: int,
	eval_seed: int,
) -> Tuple[Dict, np.ndarray, Dict[str, np.ndarray], List]:
	res_reps = defaultdict(list)
	metric_reps = defaultdict(list)
	rng = np.random.default_rng(eval_seed)
	seeds = rng.choice(10000, size=eval_repeat, replace=False)
	for seed in seeds:
		init_state = _build_sweeping_eval_init_state(agent, rng) if isinstance(agent, SweepingAgent) else None
		eval_res = eval_on_all_sessions(
			env=env,
			agent=agent,
			seed=int(seed),
			init_state=init_state,
		)
		res_reps['reward'].append(eval_res['all_reward'])
		for key, values in eval_res['all_metrics'].items():
			metric_reps[key].append(values)
		res_reps['trajectories'].append(eval_res['all_trajectories'])

	reward_reps = np.array(res_reps['reward'])
	metric_reps = {key: np.array(values) for key, values in metric_reps.items()}
	stats = {
		'avg_reward': float(reward_reps.mean(axis=1).mean()),
		'std_reward': float(reward_reps.mean(axis=1).std()),
		'avg_reward_per_session': reward_reps.mean(axis=0).tolist(),
		'std_reward_per_session': reward_reps.std(axis=0).tolist(),
	}
	for key, values in metric_reps.items():
		stats[f'avg_{key}'] = float(values.mean(axis=1).mean())
		stats[f'std_{key}'] = float(values.mean(axis=1).std())
		stats[f'avg_{key}_per_session'] = values.mean(axis=0).tolist()
		stats[f'std_{key}_per_session'] = values.std(axis=0).tolist()

	return stats, reward_reps, metric_reps, res_reps['trajectories']


def evaluate_trivial_monitoring_agent(
	agent_type: str,
	num_drones: int,
	log_dir: str,
	eval_repeat: int,
	eval_seed: int,
	norm_obs: bool = True,
	reward_type: str = 'congestion',
) -> Dict:
	log_path = Path(log_dir)
	log_path.mkdir(parents=True, exist_ok=True)

	trainset, testset = initialize_dataset()
	env: TrafficMonitorEnv = build_traffic_monitor_env(
		trainset=trainset,
		testset=testset,
		num_drones=num_drones,
		env_type='test',
		norm_obs=norm_obs,
		reward_type=reward_type,
	)
	agent = build_trivial_monitoring_agent(agent_type, num_drones=num_drones, sweeping_pos=env.all_positions)

	stats, reward_reps, metric_reps, trajectories = eval_agent_repeats(
		env=env,
		agent=agent,
		eval_repeat=eval_repeat,
		eval_seed=eval_seed,
	)
	logger.info('Drone Monitoring Evaluation Results: {}'.format(stats))

	save_path = log_path / 'rep{}x_results.json'.format(eval_repeat)
	with save_path.open('w') as f:
		json.dump(
			{
				'stats': stats,
				'reward_all_repeats': reward_reps.tolist(),
				'metrics_all_repeats': {key: values.tolist() for key, values in metric_reps.items()},
				'trajectories_all_repeats': trajectories,
			},
			f,
			indent=4,
		)

	from skymonitor.visualize import animate_trajectory

	logger.info("Visualizeing the trajectories of some evaluation runs and sessions.")
	for rep, all_traj in enumerate(trajectories):
		if not rep % 5 == 0:
			continue
		for session_id, traj in enumerate(all_traj):
			if session_id % 20 == 0:
				animate_trajectory(
					grid_xy=env.map_structure.grid_xy_of_nodes,
					trajectory=traj,
					save_path=log_path / f'traj_rep{rep}_session{session_id}.gif',
				)

	env.close()
	return stats


def get_high_reward_pos(traffic_data, map_structure, k: int = 10, return_reward: bool = False) -> List[Tuple[int, int]]:
	"""Calculate the top-k most rewarding positions for all sessions and time steps."""

	state_change_score = np.logical_or(
		traffic_data.enters, traffic_data.exits
	).astype(float)  # shape (S, T, P)
	state_change_score = state_change_score[:, 1:, :]  # ignore the first time step which is not rewarded in RL
	# weight by average flow to prioritize high-traffic locations
	avg_flow_by_location = traffic_data.flow.mean(axis=(0,1))  # shape (P, )
	state_change_score = state_change_score * avg_flow_by_location[None, None, :]  # shape (S, T, P)

	total_change = state_change_score.sum(axis=(0, 1))  # shape (P, )
	grid_ids = map_structure.grid_id_of_nodes # shape (P, )
	unique_gids = np.unique(grid_ids)
	grid_rewards = [total_change[grid_ids == gid].sum() for gid in unique_gids]
	idxs = np.argsort(grid_rewards)[::-1][:k]
	rewarding_gids = unique_gids[idxs]
	grid_id_to_xy = {gid:xy for xy, gid in map_structure.grid_xy_to_id.items()}

	positions = [grid_id_to_xy[gid] for gid in rewarding_gids]

	if return_reward:
		rewards = [grid_rewards[idx] for idx in idxs]
		reward_by_session = [
			sum(
				[state_change_score[s][:, grid_ids == gid].sum() for gid in rewarding_gids]
			)
			for s in range(state_change_score.shape[0])
		]
		assert np.isclose(sum(reward_by_session), sum(rewards)), "Reward calculation mismatch."
		return positions, rewards
	else:
		return positions

def visualize_env_data():

	from stable_baselines3.common.env_checker import check_env
	from gymnasium.utils.env_checker import check_env as gym_check_env
	from skymonitor.visualize import visualize_data_as_grid

	num_drones = 10
	trainset, testset = initialize_dataset()
	train_env, test_env = build_traffic_monitor_env(trainset=trainset, testset=testset, num_drones=num_drones, env_type='both', norm_obs=True)

	# we visualize the average reward for grids, which is how much (on average) a static agent would receive
	# if it stays at the same grid in all the sessions
	for env, name in zip([train_env, test_env], ['train', 'test']):
		state_change = np.logical_or(env.traffic_data.enters, env.traffic_data.exits).astype(float)
		flow_by_location = env.traffic_data.flow.mean(axis=(0,1))  # shape (P)
		visualize_data_as_grid(
			grid_xy=env.map_structure.grid_xy_of_nodes,
			node_data=state_change.mean(axis=0).sum(axis=0),
			agg='sum',
			note=f'state_change_by_grid_{name}'
		)
		visualize_data_as_grid(
			grid_xy=env.map_structure.grid_xy_of_nodes,
			node_data=state_change.mean(axis=0).sum(axis=0) * flow_by_location,
			agg='sum',
			note=f'state_change_weighted_by_flow_{name}'
		)

	check_env(train_env, warn=True)
	check_env(test_env, warn=True)
	# This will catch many common issues
	try:
		gym_check_env(train_env)
		gym_check_env(test_env)
		print("Environment passes all checks!")
	except Exception as e:
		print(f"Environment has issues: {e}")


if __name__ == '__main__':
	from skytraffic.utils.event_logger import setup_logger
	from skytraffic.utils.io import make_dir_if_not_exist


	parser = argparse.ArgumentParser(description='Build and evaluate the traffic monitoring environment.')
	parser.add_argument(
		'--agent-type',
		type=str,
		default='random',
		choices=['random', 'static', 'sweeping'],
		help='Evaluate a trivial baseline agent and exit.',
	)
	parser.add_argument('--num-drones', type=int, default=10, help='Number of drones used for trivial-agent evaluation.')
	parser.add_argument('--logdir', type=str, default='scratch/drone_monitor_env', help='Directory to store evaluation logs and outputs.')
	parser.add_argument('--log-level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
	parser.add_argument('--eval-repeat', type=int, default=5, help='Number of evaluation runs.')
	parser.add_argument('--eval-seed', type=int, default=888, help='Seed for the evaluation environment.')
	parser.add_argument('--norm-obs', action=argparse.BooleanOptionalAction, default=True, help='Normalize observations.')
	parser.add_argument('--reward-type', type=str, default='congestion', choices=['congestion', 'mfd'], help='Reward definition for evaluation.')
	args = parser.parse_args()

	make_dir_if_not_exist(args.logdir)
	logger = setup_logger(
		name='skymonitor',
		log_file='{}/experiment.log'.format(args.logdir),
		level=getattr(logging, args.log_level.upper()),
	)
	logger.info('Arguments: {}'.format(args))

	evaluate_trivial_monitoring_agent(
			agent_type=args.agent_type,
			num_drones=args.num_drones,
			log_dir=args.logdir,
			eval_repeat=args.eval_repeat,
			eval_seed=args.eval_seed,
			norm_obs=args.norm_obs,
			reward_type=args.reward_type,
		)
