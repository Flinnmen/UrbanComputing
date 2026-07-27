"""Profile deterministic and GMM forecasting configs."""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch

from skytraffic.config import LazyConfig, instantiate


DEFAULT_CONFIGS = [
	'config/gwnet/GWNET.py',
	'config/gwnet/GWNET_GMM.py',
	'config/mtgnn/MTGNN.py',
	'config/mtgnn/MTGNN_GMM.py',
	'config/stid/STID.py',
	'config/stid/STID_GMM.py',
	'config/staeformer/STAEformer.py',
	'config/staeformer/STAEformer_GMM.py',
	'config/lgc/LSTMGCNConv.py',
	'config/lgc/LSTMGCNConv_GMM.py',
]

HEAD_NAMES = (
	'prediction_head',
	'end_conv_2',
	'regression_layer',
	'output_proj',
	'prediction',
)


def parse_args():
	parser = argparse.ArgumentParser(
		description='Profile train and inference speed for deterministic/GMM configs.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
	parser.add_argument('--steps', type=int, default=100)
	parser.add_argument('--warmup-steps', type=int, default=5)
	parser.add_argument('--device', default=None)
	return parser.parse_args()


def endless(loader):
	while True:
		yield from loader


def sync(device):
	if device.type == 'cuda':
		torch.cuda.synchronize(device)


def peak_mb(device):
	return torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None


def config_label(config_path):
	stem = Path(config_path).stem
	kind = 'gmm' if re.search(r'(^|_)GMM', stem) else 'deterministic'
	arch = re.sub(r'^(PEMSBAY|SimbarcaSpd)_', '', stem)
	return re.sub(r'_GMM.*$', '', arch), kind


def head_params(model):
	network = getattr(model, 'network', model)
	total = 0
	for name in HEAD_NAMES:
		module = getattr(network, name, None)
		if module is not None:
			total += sum(p.numel() for p in module.parameters() if p.requires_grad)
	return total


def build(config_path, args):
	cfg = LazyConfig.load(config_path)
	if args.device:
		cfg.train.device = args.device

	train_set = instantiate(cfg.dataset.train)
	cfg.dataloader.train.dataset = train_set
	cfg.dataloader.train.collate_fn = train_set.collate_fn
	loader = instantiate(cfg.dataloader.train)

	cfg.model.metadata = train_set.metadata
	model = instantiate(cfg.model).to(cfg.train.device)
	cfg.optimizer.params = model.parameters()
	return cfg, loader, model, instantiate(cfg.optimizer)


def train_steps(model, loader, optimizer, steps, device):
	model.train()
	batches = endless(loader)
	optimizer.zero_grad(set_to_none=True)

	sync(device)
	start = time.perf_counter()
	for _ in range(steps):
		losses = model(next(batches))
		loss = sum(losses.values()) if isinstance(losses, dict) else losses
		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)
	sync(device)
	return time.perf_counter() - start


def inference_steps(model, loader, steps, device):
	model.eval()
	batches = endless(loader)

	sync(device)
	start = time.perf_counter()
	with torch.no_grad():
		for _ in range(steps):
			batch = next(batches)
			# ForecastModel can skip target normalization during inference.
			if isinstance(batch, dict) and 'source' in batch:
				batch = {'source': batch['source']}
			model(batch)
	sync(device)
	return time.perf_counter() - start


def profile(config_path, args):
	cfg, loader, model, optimizer = build(config_path, args)
	device = torch.device(cfg.train.device)
	arch, kind = config_label(config_path)

	if args.warmup_steps:
		train_steps(model, loader, optimizer, args.warmup_steps, device)

	if device.type == 'cuda':
		torch.cuda.reset_peak_memory_stats(device)
	train_time = train_steps(model, loader, optimizer, args.steps, device)
	train_mem = peak_mb(device)

	if device.type == 'cuda':
		torch.cuda.reset_peak_memory_stats(device)
	infer_time = inference_steps(model, loader, args.steps, device)
	infer_mem = peak_mb(device)

	return {
		'architecture': arch,
		'kind': kind,
		'config': config_path,
		'batch': loader.batch_size,
		'head_params': head_params(model),
		'train_ms_step': train_time * 1000 / args.steps,
		'infer_ms_step': infer_time * 1000 / args.steps,
		'train_gpu_mb': train_mem,
		'infer_gpu_mb': infer_mem,
	}


def compare(results):
	rows = []
	for arch in sorted({r['architecture'] for r in results}):
		det = next((r for r in results if r['architecture'] == arch and r['kind'] == 'deterministic'), None)
		if det is None:
			continue
		for gmm in [r for r in results if r['architecture'] == arch and r['kind'] == 'gmm']:
			rows.append(
				{
					'architecture': arch,
					'gmm_config': gmm['config'],
					'train_x_det': gmm['train_ms_step'] / det['train_ms_step'],
					'infer_x_det': gmm['infer_ms_step'] / det['infer_ms_step'],
					'head_params_x_det': gmm['head_params'] / det['head_params'],
					'train_gpu_delta_mb': (
						gmm['train_gpu_mb'] - det['train_gpu_mb'] if gmm['train_gpu_mb'] is not None else None
					),
					'infer_gpu_delta_mb': (
						gmm['infer_gpu_mb'] - det['infer_gpu_mb'] if gmm['infer_gpu_mb'] is not None else None
					),
				}
			)
	return rows


def print_latex_table(title, label, rows, columns):
	print(f'\n{title}')
	table = pd.DataFrame(rows, columns=columns)
	print(
		table.to_latex(
			index=False,
			caption=title,
			label=label,
			escape=True,
			na_rep='-',
			float_format=lambda value: f'{value:.3f}',
		)
	)


def main():
	args = parse_args()
	results = []
	for config_path in args.configs:
		print(f'Profiling {config_path} ...', flush=True)
		results.append(profile(config_path, args))

	config_cols = [
		'architecture',
		'kind',
		'config',
		'batch',
		'head_params',
		'train_ms_step',
		'infer_ms_step',
		'train_gpu_mb',
		'infer_gpu_mb',
	]
	comparison_cols = [
		'architecture',
		'gmm_config',
		'train_x_det',
		'infer_x_det',
		'head_params_x_det',
		'train_gpu_delta_mb',
		'infer_gpu_delta_mb',
	]
	comparisons = compare(results)
	print_latex_table('Per-config results', 'tab:gmm-profile-configs', results, config_cols)
	if comparisons:
		print_latex_table(
			'GMM vs deterministic by architecture',
			'tab:gmm-profile-comparisons',
			comparisons,
			comparison_cols,
		)


if __name__ == '__main__':
	main()
