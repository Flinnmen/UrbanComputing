"""Summarize SkyMonitor evaluation JSON files under scratch."""
import argparse
import json
import re
import statistics
from pathlib import Path


METRICS_BY_REWARD_TYPE = {
	'mfd': [
		('Flow Abs Error', 'mfd_flow_abs_error'),
		('Flow Rel Error', 'mfd_flow_rel_error'),
		('Density Abs Error', 'mfd_density_abs_error'),
		('Density Rel Error', 'mfd_density_rel_error'),
		('MFD Error', 'mfd_error'),
	],
	'congestion': [
		('Reward', 'reward'),
		('Covered Change', 'covered_congestion_change'),
		('Total Change', 'total_congestion_change'),
		('Coverage Rate', 'congestion_change_coverage_rate'),
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


def classify_run(run_dir: Path, reward_type: str):
	prefix = 'drone_monitor_'
	if not run_dir.name.startswith(prefix):
		return None

	name = run_dir.name.removeprefix(prefix)
	if name == f'random_{reward_type}':
		return {'kind': 'baseline', 'label': name, 'sort_key': (0, 0)}
	if name == f'static_{reward_type}':
		return {'kind': 'baseline', 'label': name, 'sort_key': (1, 0)}
	if name == f'sweeping_{reward_type}':
		return {'kind': 'baseline', 'label': name, 'sort_key': (1, 1)}

	match = re.fullmatch(rf'ppo_(.+)_{re.escape(reward_type)}_(\d+)', name)
	if match:
		policy, index = match.group(1), int(match.group(2))
		group_label = f'ppo_{policy}_{reward_type}'
		return {
			'kind': 'ppo',
			'label': name,
			'group_label': group_label,
			'index': index,
			'sort_key': (2, group_label, index),
		}

	return None


def load_runs(scratch_dir: Path, reward_type: str, result_glob: str):
	runs = []
	for run_dir in scratch_dir.iterdir():
		if not run_dir.is_dir():
			continue
		run_info = classify_run(run_dir, reward_type)
		if run_info is None:
			continue
		result_file = pick_result_file(run_dir, result_glob)
		if result_file is None:
			continue
		with result_file.open() as f:
			run_info['stats'] = json.load(f)['stats']
		run_info['result_file'] = result_file
		runs.append(run_info)

	return sorted(runs, key=lambda run: run['sort_key'])


def value_from_stats(stats: dict, key: str) -> tuple[float, float]:
	return stats[f'avg_{key}'], stats[f'std_{key}']


def summarize_group(runs: list[dict], group_label: str, metrics: list[tuple[str, str]]) -> dict:
	indices = sorted(run['index'] for run in runs)
	if len(indices) == 1:
		index_label = str(indices[0])
	elif indices == list(range(indices[0], indices[-1] + 1)):
		index_label = f'{indices[0]}-{indices[-1]}'
	else:
		index_label = ','.join(str(index) for index in indices)
	label = f'{group_label}_{index_label}'
	stats = {}
	for _, key in metrics:
		values = [run['stats'][f'avg_{key}'] for run in runs]
		stats[f'avg_{key}'] = statistics.fmean(values)
		stats[f'std_{key}'] = statistics.pstdev(values)
	return {'label': label, 'stats': stats}


def format_value(avg: float, std: float, plus_minus: str) -> str:
	return f'{avg:.6g} {plus_minus} {std:.3g}'


def print_markdown_table(rows: list[dict], metrics: list[tuple[str, str]], plus_minus: str) -> None:
	headers = ['Run'] + [label for label, _ in metrics]
	print('| ' + ' | '.join(headers) + ' |')
	print('|---|' + '|'.join(['---:'] * len(metrics)) + '|')
	for row in rows:
		values = [
			format_value(*value_from_stats(row['stats'], key), plus_minus)
			for _, key in metrics
		]
		print('| ' + row['label'] + ' | ' + ' | '.join(values) + ' |')


def build_rows(runs: list[dict], show_ppo_runs: bool, metrics: list[tuple[str, str]]) -> list[dict]:
	rows = [run for run in runs if run['kind'] == 'baseline']
	ppo_runs = [run for run in runs if run['kind'] == 'ppo']
	if show_ppo_runs:
		rows.extend(ppo_runs)

	grouped = {}
	for run in ppo_runs:
		grouped.setdefault(run['group_label'], []).append(run)
	for group_label in sorted(grouped):
		rows.append(summarize_group(grouped[group_label], group_label, metrics))

	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description='Summarize SkyMonitor evaluation results.')
	parser.add_argument('--scratch-dir', type=Path, default=Path('scratch'))
	parser.add_argument('--reward-type', type=str, default='mfd', choices=sorted(METRICS_BY_REWARD_TYPE))
	parser.add_argument('--result-glob', type=str, default='rep*x_results.json')
	parser.add_argument('--show-ppo-runs', action='store_true', help='Also print each PPO run before the grouped PPO summary.')
	parser.add_argument('--ascii', action='store_true', help='Use "+/-" instead of the "±" symbol.')
	args = parser.parse_args()
	metrics = METRICS_BY_REWARD_TYPE[args.reward_type]
	plus_minus = '+/-' if args.ascii else '±'

	runs = load_runs(args.scratch_dir, args.reward_type, args.result_glob)
	if not runs:
		raise FileNotFoundError(f'No matching result files found under {args.scratch_dir}.')

	print_markdown_table(
		build_rows(runs, show_ppo_runs=args.show_ppo_runs, metrics=metrics),
		metrics,
		plus_minus,
	)


if __name__ == '__main__':
	main()
