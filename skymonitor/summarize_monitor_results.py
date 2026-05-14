"""Summarize SkyMonitor evaluation JSON files under scratch."""
import argparse
import json
import re
from pathlib import Path


REWARD_TYPES = ['congestion', 'mfd', 'congestion_level', 'region_mfd']

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
	return (2, label)


def load_methods(scratch_dir: Path, result_glob: str):
	methods = []
	for run_dir in scratch_dir.iterdir():
		if not run_dir.is_dir():
			continue
		result_file = pick_result_file(run_dir, result_glob)
		if result_file is None:
			continue
		with result_file.open() as f:
			stats = json.load(f)['stats']
		methods.append({
			'label': run_dir.name.removeprefix('drone_monitor_'),
			'stats': stats,
			'result_file': result_file,
			'sort_key': method_sort_key(run_dir.name),
		})

	return sorted(methods, key=lambda method: method['sort_key'])


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


def render_summary(methods: list[dict], plus_minus: str) -> str:
	lines = ['# SkyMonitor Evaluation Summary', '']
	for eval_type in REWARD_TYPES:
		metrics = discover_region_metrics(methods) if eval_type == 'region_mfd' else TABLE_METRICS[eval_type]
		lines.append(f'## {eval_type}')
		lines.extend(render_markdown_table(methods, metrics, plus_minus))
		lines.append('')
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
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(render_summary(methods, plus_minus), encoding='utf-8')
	print(f'Saved summary to {output_path}')


if __name__ == '__main__':
	main()
