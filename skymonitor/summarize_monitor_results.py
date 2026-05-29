"""Summarize SkyMonitor evaluation JSON files under scratch."""
import argparse
import csv
import json
import re
from pathlib import Path


REWARD_TYPES = ['congestion', 'mfd', 'congestion_level', 'region_mfd']
COVERAGE_REWARD_TYPES = ['congestion', 'congestion_level', 'mfd', 'region_mfd', 'balanced']
COVERAGE_DISCARD_STEPS = 5

TABLE_METRICS = {
	'congestion': [
		('Covered', 'covered_congestion_change'),
		('Total', 'total_congestion_change'),
		('Rate', 'congestion_change_coverage_rate'),
	],
	'mfd': [
		('Flow AE', 'mfd_flow_abs_error'),
		('Flow RE', 'mfd_flow_rel_error'),
		('Density AE', 'mfd_density_abs_error'),
		('Density RE', 'mfd_density_rel_error'),
		('MFD Err', 'mfd_error'),
	],
	'congestion_level': [
		('Covered', 'covered_congestion_level_change'),
		('Total', 'total_congestion_level_change'),
		('W Covered', 'covered_weighted_congestion_level_change'),
		('W Total', 'total_weighted_congestion_level_change'),
		('Rate', 'congestion_level_change_coverage_rate'),
		('W Rate', 'weighted_congestion_level_change_coverage_rate'),
	],
}


def pick_result_file(run_dir: Path, result_glob: str) -> Path | None:
	files = list(run_dir.glob(result_glob))
	if not files:
		return None

	def repeat_count(path: Path) -> int:
		match = re.fullmatch(r'rep(\d+)x_results\.json', path.name)
		return int(match.group(1)) if match else -1

	return max(files, key=lambda path: (repeat_count(path), path.name))


def method_sort_key(name: str):
	label = name.removeprefix('drone_monitor_')
	baseline_order = {'random': 0, 'static': 1, 'sweeping': 2}
	for baseline, order in baseline_order.items():
		if label == baseline or label.startswith(f'{baseline}_'):
			return (0, order, label)

	match = re.fullmatch(r'ppo_(.+)_(\d+)', label)
	if match:
		return (1, match.group(1), int(match.group(2)))
	if label.startswith('ppo_'):
		return (1, label.removeprefix('ppo_'), 0)
	return (2, label)


def ppo_group_label(label: str) -> str:
	match = re.fullmatch(r'(ppo_.+)_(\d+)', label)
	return match.group(1) if match else label


def grouped_std(values: list[float], avg: float) -> float:
	return (sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5


def aggregate_methods(label: str, methods: list[dict]) -> dict:
	if len(methods) == 1:
		return methods[0]

	stats = {}
	avg_keys = sorted({
		key
		for method in methods
		for key, value in method['stats'].items()
		if key.startswith('avg_') and isinstance(value, (int, float))
	})
	for avg_key in avg_keys:
		values = [method['stats'][avg_key] for method in methods if avg_key in method['stats']]
		if len(values) != len(methods):
			continue
		avg = sum(values) / len(values)
		stats[avg_key] = avg
		stats[f'std_{avg_key.removeprefix("avg_")}'] = grouped_std(values, avg)

	return {
		'label': label,
		'stats': stats,
		'result_file': [method['result_file'] for method in methods],
		'sort_key': method_sort_key(label),
	}


def load_methods(scratch_dir: Path, result_glob: str):
	method_groups = {}
	for run_dir in scratch_dir.iterdir():
		if not run_dir.is_dir():
			continue
		result_file = pick_result_file(run_dir, result_glob)
		if result_file is None:
			continue
		with result_file.open() as f:
			stats = json.load(f)['stats']
		label = run_dir.name.removeprefix('drone_monitor_')
		group_label = ppo_group_label(label)
		method_groups.setdefault(group_label, []).append({
			'label': label,
			'stats': stats,
			'result_file': result_file,
			'sort_key': method_sort_key(run_dir.name),
		})

	methods = [
		aggregate_methods(label, group)
		for label, group in method_groups.items()
	]
	return sorted(methods, key=lambda method: method['sort_key'])


def infer_ppo_reward_type(run_dir: Path) -> str | None:
	match = re.fullmatch(r'drone_monitor_ppo_[^_]+_(.+)_(\d+)', run_dir.name)
	if match is None:
		return None
	reward_type = match.group(1)
	return reward_type if reward_type in COVERAGE_REWARD_TYPES else None


def load_coverage_runs(scratch_dir: Path, result_glob: str) -> dict[str, list[dict]]:
	runs_by_reward = {reward_type: [] for reward_type in COVERAGE_REWARD_TYPES}
	for run_dir in scratch_dir.iterdir():
		if not run_dir.is_dir():
			continue
		reward_type = infer_ppo_reward_type(run_dir)
		if reward_type is None:
			continue
		result_file = pick_result_file(run_dir, result_glob)
		if result_file is None:
			continue
		with result_file.open() as fp:
			result = json.load(fp)
		if 'trajectories_all_repeats' not in result:
			continue
		runs_by_reward[reward_type].append({
			'result_file': result_file,
			'trajectories': result['trajectories_all_repeats'],
		})
	return {reward_type: runs for reward_type, runs in runs_by_reward.items() if runs}


def trajectory_support(runs_by_reward: dict[str, list[dict]]) -> set[tuple[int, int]]:
	support = set()
	for runs in runs_by_reward.values():
		for run in runs:
			for repeat in run['trajectories']:
				for session in repeat:
					for frame in session:
						support.update((int(pos[0]), int(pos[1])) for pos in frame)
	return support


def coverage_maps_from_runs(
	runs_by_reward: dict[str, list[dict]],
	discard_steps: int,
) -> tuple[list[str], dict[str, object], list[tuple[int, int]]]:
	import numpy as np

	support = trajectory_support(runs_by_reward)
	if not support:
		raise ValueError('No trajectory positions found for coverage analysis.')

	width = max(x for x, _ in support) + 1
	height = max(y for _, y in support) + 1
	support_order = [(x, y) for y in range(height) for x in range(width) if (x, y) in support]
	support_index = {pos: idx for idx, pos in enumerate(support_order)}
	reward_types = [reward_type for reward_type in COVERAGE_REWARD_TYPES if reward_type in runs_by_reward]
	coverage = {}

	for reward_type in reward_types:
		counts = np.zeros(len(support_order), dtype=np.float64)
		frame_count = 0
		for run in runs_by_reward[reward_type]:
			for repeat in run['trajectories']:
				for session in repeat:
					for frame in session[discard_steps:]:
						frame_count += 1
						for pos in {tuple(map(int, item)) for item in frame}:
							if pos in support_index:
								counts[support_index[pos]] += 1.0
		if frame_count == 0:
			raise ValueError(f'No trajectory frames remain after discarding {discard_steps} steps for {reward_type}.')

		vector = counts / frame_count
		grid = np.full((height, width), np.nan, dtype=np.float64)
		for (x, y), value in zip(support_order, vector):
			grid[y, x] = value
		coverage[reward_type] = {
			'grid': grid,
			'vector': vector,
			'frame_count': frame_count,
			'run_count': len(runs_by_reward[reward_type]),
		}

	return reward_types, coverage, support_order


def write_matrix_csv(labels: list[str], matrix, output_path: Path) -> None:
	with output_path.open('w', newline='') as fp:
		writer = csv.writer(fp)
		writer.writerow(['reward_type', *labels])
		for label, row in zip(labels, matrix):
			writer.writerow([label, *[f'{value:.6g}' for value in row]])


def plot_coverage_heatmaps(reward_types: list[str], coverage: dict[str, object], output_path: Path) -> None:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib import cm
	import numpy as np

	try:
		import seaborn as sns
		sns.set_theme(style='darkgrid')
	except ImportError:
		pass

	vmax = max(float(np.nanmax(coverage[reward_type]['grid'])) for reward_type in reward_types)
	cmap = cm.turbo.copy()
	cmap.set_bad(color='white')
	fig, axes = plt.subplots(1, len(reward_types), figsize=(4 * len(reward_types), 4.5), squeeze=False)

	for ax, reward_type in zip(axes[0], reward_types):
		grid = coverage[reward_type]['grid']
		height, width = grid.shape
		image = ax.imshow(
			grid,
			origin='lower',
			cmap=cmap,
			vmin=0.0,
			vmax=vmax,
			extent=(-0.5, width - 0.5, -0.5, height - 0.5),
			interpolation='none',
		)
		for x in range(width + 1):
			ax.axvline(x - 0.5, color='gray', linewidth=0.4, alpha=0.35)
		for y in range(height + 1):
			ax.axhline(y - 0.5, color='gray', linewidth=0.4, alpha=0.35)
		ax.set_title(reward_type)
		ax.set_xlabel('Grid X')
		ax.set_ylabel('Grid Y')
		ax.set_aspect('equal', adjustable='box')

	fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label='Average coverage rate')
	fig.savefig(output_path, bbox_inches='tight')
	plt.close(fig)


def plot_correlation_heatmap(labels: list[str], matrix, output_path: Path) -> None:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	try:
		import seaborn as sns
		sns.set_theme(style='darkgrid')
	except ImportError:
		pass

	fig, ax = plt.subplots(figsize=(6, 5))
	image = ax.imshow(matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
	ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha='right')
	ax.set_yticks(range(len(labels)), labels=labels)
	for row_idx, row in enumerate(matrix):
		for col_idx, value in enumerate(row):
			ax.text(col_idx, row_idx, f'{value:.2f}', ha='center', va='center', fontsize=9)
	ax.set_title('Coverage Correlation')
	fig.colorbar(image, ax=ax, shrink=0.8, label='Pearson r')
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches='tight')
	plt.close(fig)


def write_coverage_analysis(
	scratch_dir: Path,
	result_glob: str,
	output_dir: Path,
) -> dict | None:
	import numpy as np

	runs_by_reward = load_coverage_runs(scratch_dir, result_glob)
	if not runs_by_reward:
		return None

	output_dir.mkdir(parents=True, exist_ok=True)
	reward_types, coverage, support_order = coverage_maps_from_runs(runs_by_reward, COVERAGE_DISCARD_STEPS)
	vectors = np.vstack([coverage[reward_type]['vector'] for reward_type in reward_types])
	correlation = np.corrcoef(vectors)

	heatmap_path = output_dir / 'average_coverage_heatmaps.pdf'
	correlation_plot_path = output_dir / 'coverage_correlation_heatmap.pdf'
	correlation_csv_path = output_dir / 'coverage_correlation.csv'

	plot_coverage_heatmaps(reward_types, coverage, heatmap_path)
	plot_correlation_heatmap(reward_types, correlation, correlation_plot_path)
	write_matrix_csv(reward_types, correlation, correlation_csv_path)

	return {
		'reward_types': reward_types,
		'correlation': correlation.tolist(),
		'output_dir': output_dir,
		'heatmap_path': heatmap_path,
		'correlation_plot_path': correlation_plot_path,
		'correlation_csv_path': correlation_csv_path,
		'discard_steps': COVERAGE_DISCARD_STEPS,
		'grid_cell_count': len(support_order),
		'run_counts': {reward_type: coverage[reward_type]['run_count'] for reward_type in reward_types},
		'frame_counts': {reward_type: coverage[reward_type]['frame_count'] for reward_type in reward_types},
	}


def value_from_stats(stats: dict, key: str) -> tuple[float, float] | None:
	if f'avg_{key}' not in stats or f'std_{key}' not in stats:
		return None
	return stats[f'avg_{key}'], stats[f'std_{key}']


def discover_region_metrics(methods: list[dict]) -> list[tuple[str, str]]:
	region_ids = set()
	for method in methods:
		for stat_key in method['stats']:
			match = re.fullmatch(r'avg_region_(\d+)_mfd_error', stat_key)
			if match and f'std_region_{match.group(1)}_mfd_error' in method['stats']:
				region_ids.add(int(match.group(1)))

	metrics = [('Avg Err', 'region_mfd_error')]
	metrics.extend((f'R{region_id}', f'region_{region_id}_mfd_error') for region_id in sorted(region_ids))
	return metrics


def format_value(avg: float, std: float, plus_minus: str) -> str:
	return f'{avg:.6g} {plus_minus} {std:.3g}'


def render_markdown_table(methods: list[dict], metrics: list[tuple[str, str]], plus_minus: str) -> list[str]:
	lines = []
	headers = ['Method'] + [label for label, _ in metrics]
	lines.append('| ' + ' | '.join(headers) + ' |')
	lines.append('|---|' + '|'.join(['---:'] * len(metrics)) + '|')
	for method in methods:
		values = []
		for _, key in metrics:
			value = value_from_stats(method['stats'], key)
			values.append('n/a' if value is None else format_value(*value, plus_minus))
		lines.append('| ' + method['label'] + ' | ' + ' | '.join(values) + ' |')
	return lines


def render_correlation_table(labels: list[str], matrix: list[list[float]]) -> list[str]:
	lines = []
	lines.append('| Reward type | ' + ' | '.join(labels) + ' |')
	lines.append('|---|' + '|'.join(['---:'] * len(labels)) + '|')
	for label, row in zip(labels, matrix):
		lines.append('| ' + label + ' | ' + ' | '.join(f'{value:.4f}' for value in row) + ' |')
	return lines


def render_coverage_summary(coverage_summary: dict | None) -> list[str]:
	if coverage_summary is None:
		return []
	lines = [
		'## spatial_coverage',
		f'Discarded first {coverage_summary["discard_steps"]} trajectory steps. '
		f'Correlation uses {coverage_summary["grid_cell_count"]} grid cells.',
		'',
		'Artifacts:',
		f'- Coverage heatmaps: `{coverage_summary["heatmap_path"]}`',
		f'- Coverage correlation heatmap: `{coverage_summary["correlation_plot_path"]}`',
		f'- Coverage correlation CSV: `{coverage_summary["correlation_csv_path"]}`',
		'',
		'Run counts by reward type: '
		+ ', '.join(
			f'{reward_type}={coverage_summary["run_counts"][reward_type]}'
			for reward_type in coverage_summary['reward_types']
		),
		'Frame counts after discard: '
		+ ', '.join(
			f'{reward_type}={coverage_summary["frame_counts"][reward_type]}'
			for reward_type in coverage_summary['reward_types']
		),
		'',
	]
	lines.extend(render_correlation_table(coverage_summary['reward_types'], coverage_summary['correlation']))
	lines.append('')
	return lines


def render_summary(methods: list[dict], plus_minus: str, coverage_summary: dict | None = None) -> str:
	lines = ['# SkyMonitor Evaluation Summary', '']
	for eval_type in REWARD_TYPES:
		metrics = discover_region_metrics(methods) if eval_type == 'region_mfd' else TABLE_METRICS[eval_type]
		lines.append(f'## {eval_type}')
		lines.extend(render_markdown_table(methods, metrics, plus_minus))
		lines.append('')
	lines.extend(render_coverage_summary(coverage_summary))
	return '\n'.join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(description='Summarize SkyMonitor evaluation results.')
	parser.add_argument('--scratch-dir', type=Path, default=Path('scratch'))
	parser.add_argument('--result-glob', type=str, default='rep*x_results.json')
	parser.add_argument('--output', type=Path, default=None, help='Markdown file to write.')
	parser.add_argument('--ascii', action='store_true', help='Use "+/-" instead of the "±" symbol.')
	args = parser.parse_args()
	plus_minus = '+/-' if args.ascii else '±'

	methods = load_methods(args.scratch_dir, args.result_glob)
	if not methods:
		raise FileNotFoundError(f'No matching result files found under {args.scratch_dir}.')

	output_path = args.output if args.output is not None else args.scratch_dir / 'monitor_results_summary.md'
	coverage_summary = write_coverage_analysis(
		scratch_dir=args.scratch_dir,
		result_glob=args.result_glob,
		output_dir=output_path.parent / 'monitor_coverage',
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(render_summary(methods, plus_minus, coverage_summary), encoding='utf-8')
	print(f'Saved summary to {output_path}')
	if coverage_summary is not None:
		print(f'Saved coverage analysis to {coverage_summary["output_dir"]}')


if __name__ == '__main__':
	main()
