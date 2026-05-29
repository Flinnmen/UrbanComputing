"""Plot SkyMonitor balanced-reward weight experiments."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='darkgrid')

RAW_METRICS = (
	'mfd_error',
	'region_mfd_error',
	'congestion_change_coverage_rate',
	'weighted_congestion_level_change_coverage_rate',
)
DERIVED_METRICS = ('mfd_group_error', 'congestion_group_coverage')
ALL_METRICS = RAW_METRICS + DERIVED_METRICS
WEIGHT_KEYS = ('congestion', 'congestion_level', 'mfd', 'region_mfd')

AXIS_LABELS = {
	'mfd_error': 'Single-region MFD error',
	'region_mfd_error': 'Multi-region MFD error',
	'congestion_change_coverage_rate': 'Congestion state-change coverage rate',
	'weighted_congestion_level_change_coverage_rate': 'Weighted multi-level congestion coverage rate',
	'mfd_group_error': 'Mean MFD error',
	'congestion_group_coverage': 'Mean congestion coverage rate',
}


def read_result_rows(scratch_dir: Path) -> list[dict]:
	rows = []
	for result_file in scratch_dir.glob('**/rep*x_results.json'):
		rel_parts = result_file.relative_to(scratch_dir).parts
		if len(rel_parts) < 4:
			continue
		task, drone_part, label, repeat = rel_parts[:4]
		if task not in {'task1', 'task2', 'task3', 'task4'}:
			continue
		match = re.fullmatch(r'dr(\d+)', drone_part)
		if match is None:
			continue

		with result_file.open() as fp:
			result = json.load(fp)
		stats = result['stats']
		weights = result.get('config', {}).get('balanced_reward_weights')
		if weights is None:
			raise ValueError(f'{result_file} does not include balanced_reward_weights in config.')

		row = {
			'task': task,
			'num_drones': int(match.group(1)),
			'label': label,
			'repeat': repeat,
			'result_file': str(result_file),
		}
		for key in WEIGHT_KEYS:
			row[key] = float(weights[key])
		for metric in RAW_METRICS:
			stat_key = f'avg_{metric}'
			if stat_key in stats:
				row[metric] = float(stats[stat_key])
		rows.append(row)
	return rows


def metric_value(row: dict, metric: str) -> float | None:
	if metric in row:
		return row[metric]
	if metric == 'mfd_group_error':
		if 'mfd_error' in row and 'region_mfd_error' in row:
			return 0.5 * (row['mfd_error'] + row['region_mfd_error'])
	if metric == 'congestion_group_coverage':
		if 'congestion_change_coverage_rate' in row and 'weighted_congestion_level_change_coverage_rate' in row:
			return 0.5 * (
				row['congestion_change_coverage_rate']
				+ row['weighted_congestion_level_change_coverage_rate']
			)
	return None


def aggregate_rows(rows: list[dict]) -> list[dict]:
	grouped = defaultdict(list)
	for row in rows:
		key = (
			row['task'],
			row['num_drones'],
			row['label'],
			tuple(row[weight_key] for weight_key in WEIGHT_KEYS),
		)
		grouped[key].append(row)

	groups = []
	for (task, num_drones, label, weights), group_rows in grouped.items():
		group = {
			'task': task,
			'num_drones': num_drones,
			'label': label,
			'repeat_count': len(group_rows),
		}
		for key, value in zip(WEIGHT_KEYS, weights):
			group[key] = value
		for metric in ALL_METRICS:
			values = [metric_value(row, metric) for row in group_rows]
			values = [value for value in values if value is not None]
			if values:
				group[f'{metric}_mean'] = float(np.mean(values))
				group[f'{metric}_std'] = float(np.std(values))
		groups.append(group)
	return groups


def is_baseline(group: dict) -> bool:
	return all(np.isclose(group[key], 0.25) for key in WEIGHT_KEYS)


def sort_by_weight(groups: list[dict], key: str) -> list[dict]:
	return sorted(groups, key=lambda group: (group[key], group['label']))


def plot_tradeoff(
	groups: list[dict],
	x_metric: str,
	y_metric: str,
	title: str,
	output_path: Path,
	sort_weight: str,
	annotation_weight: str | None = None,
) -> None:
	groups = [
		group for group in groups
		if f'{x_metric}_mean' in group and f'{y_metric}_mean' in group
	]
	if not groups:
		return
	groups = sort_by_weight(groups, sort_weight)

	fig, ax = plt.subplots(figsize=(7, 5))
	xs = [group[f'{x_metric}_mean'] for group in groups]
	ys = [group[f'{y_metric}_mean'] for group in groups]
	xerr = [group.get(f'{x_metric}_std', 0.0) for group in groups]
	yerr = [group.get(f'{y_metric}_std', 0.0) for group in groups]
	ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker='o', linestyle='-', capsize=3)

	if annotation_weight is not None:
		for group, x, y in zip(groups, xs, ys):
			label = 'baseline' if is_baseline(group) else f'{annotation_weight}={group[annotation_weight]:.2f}'
			ax.annotate(label, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)

	ax.set_xlabel(AXIS_LABELS[x_metric])
	ax.set_ylabel(AXIS_LABELS[y_metric])
	ax.set_title(title)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches='tight')
	plt.close(fig)


def plot_observation_budget(groups: list[dict], output_path: Path) -> None:
	groups = [
		group for group in groups
		if f'mfd_group_error_mean' in group and f'congestion_group_coverage_mean' in group
	]
	if not groups:
		return

	fig, ax = plt.subplots(figsize=(7, 5))
	for num_drones in sorted({group['num_drones'] for group in groups}):
		drone_groups = sort_by_weight(
			[group for group in groups if group['num_drones'] == num_drones],
			'congestion',
		)
		xs = [group['mfd_group_error_mean'] for group in drone_groups]
		ys = [group['congestion_group_coverage_mean'] for group in drone_groups]
		xerr = [group.get('mfd_group_error_std', 0.0) for group in drone_groups]
		yerr = [group.get('congestion_group_coverage_std', 0.0) for group in drone_groups]
		ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker='o', linestyle='-', capsize=3, label=f'{num_drones} drones')

	ax.set_xlabel(AXIS_LABELS['mfd_group_error'])
	ax.set_ylabel(AXIS_LABELS['congestion_group_coverage'])
	ax.set_title('Task 4: observation budget')
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches='tight')
	plt.close(fig)


def write_csv(groups: list[dict], output_path: Path) -> None:
	fieldnames = ['task', 'num_drones', 'label', 'repeat_count', *WEIGHT_KEYS]
	for metric in ALL_METRICS:
		fieldnames.extend([f'{metric}_mean', f'{metric}_std'])
	with output_path.open('w', newline='') as fp:
		writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
		writer.writeheader()
		for group in sorted(groups, key=lambda item: (item['task'], item['num_drones'], item['label'])):
			writer.writerow(group)


def main() -> None:
	parser = argparse.ArgumentParser(description='Plot balanced reward-weight experiment results.')
	parser.add_argument('--scratch-dir', type=Path, default=Path('scratch/skymonitor_reward_weights'))
	parser.add_argument('--output-dir', type=Path, default=None)
	args = parser.parse_args()

	output_dir = args.output_dir if args.output_dir is not None else args.scratch_dir / 'figures'
	output_dir.mkdir(parents=True, exist_ok=True)

	rows = read_result_rows(args.scratch_dir)
	if not rows:
		raise FileNotFoundError(f'No reward-weight experiment result files found under {args.scratch_dir}.')
	groups = aggregate_rows(rows)
	write_csv(groups, output_dir / 'reward_weight_points.csv')

	plot_tradeoff(
		[group for group in groups if group['task'] == 'task1' and group['num_drones'] == 10],
		x_metric='mfd_error',
		y_metric='region_mfd_error',
		title='Task 1: single-region vs multi-region MFD',
		output_path=output_dir / 'task1_mfd_relation.pdf',
		sort_weight='mfd',
		annotation_weight='mfd',
	)
	plot_tradeoff(
		[group for group in groups if group['task'] == 'task2' and group['num_drones'] == 10],
		x_metric='congestion_change_coverage_rate',
		y_metric='weighted_congestion_level_change_coverage_rate',
		title='Task 2: congestion state change vs multi-level congestion',
		output_path=output_dir / 'task2_congestion_relation.pdf',
		sort_weight='congestion',
		annotation_weight='congestion',
	)
	plot_tradeoff(
		[group for group in groups if group['task'] == 'task3' and group['num_drones'] == 10],
		x_metric='mfd_group_error',
		y_metric='congestion_group_coverage',
		title='Task 3: MFD vs congestion trade-off',
		output_path=output_dir / 'task3_tradeoff_dr10.pdf',
		sort_weight='congestion',
		annotation_weight='congestion',
	)
	plot_observation_budget(
		[group for group in groups if group['task'] == 'task4' or (group['task'] == 'task3' and group['num_drones'] == 10)],
		output_dir / 'task4_observation_budget.pdf',
	)
	print(f'Saved reward-weight plots and CSV to {output_dir}')


if __name__ == '__main__':
	main()
