import numpy as np


def estimate_mfd(
	flow: np.ndarray,
	density: np.ndarray,
	segment_lengths: np.ndarray,
	coverage_mask: np.ndarray | None = None,
) -> tuple[float, float]:
	"""Estimate normalized MFD distance and time from segment-level data."""
	weights = np.asarray(segment_lengths, dtype=np.float64)
	if coverage_mask is not None:
		weights = weights * np.asarray(coverage_mask, dtype=np.float64)

	total_length = weights.sum()
	if total_length <= 0:
		return 0.0, 0.0

	flow = np.asarray(flow, dtype=np.float64)
	density = np.asarray(density, dtype=np.float64)
	return (
		float(np.dot(flow, weights) / total_length),
		float(np.dot(density, weights) / total_length),
	)


def calculate_mfd_reward(
	flow: np.ndarray,
	density: np.ndarray,
	segment_lengths: np.ndarray,
	coverage_mask: np.ndarray,
	eps: float = 1e-8,
) -> float:
	return calculate_mfd_metrics(flow, density, segment_lengths, coverage_mask, eps=eps)['reward']


def calculate_mfd_metrics(
	flow: np.ndarray,
	density: np.ndarray,
	segment_lengths: np.ndarray,
	coverage_mask: np.ndarray,
	eps: float = 1e-8,
) -> dict[str, float]:
	true_flow, true_density = estimate_mfd(flow, density, segment_lengths)
	est_flow, est_density = estimate_mfd(flow, density, segment_lengths, coverage_mask)

	flow_abs_error = abs(est_flow - true_flow)
	density_abs_error = abs(est_density - true_density)
	flow_rel_error = flow_abs_error / max(abs(true_flow), eps)
	density_rel_error = density_abs_error / max(abs(true_density), eps)
	mfd_error = 0.5 * (flow_rel_error + density_rel_error)

	return {
		'mfd_flow_true': float(true_flow),
		'mfd_flow_estimate': float(est_flow),
		'mfd_flow_abs_error': float(flow_abs_error),
		'mfd_flow_rel_error': float(flow_rel_error),
		'mfd_density_true': float(true_density),
		'mfd_density_estimate': float(est_density),
		'mfd_density_abs_error': float(density_abs_error),
		'mfd_density_rel_error': float(density_rel_error),
		'mfd_error': float(mfd_error),
		'reward': float(-mfd_error),
	}


def calculate_region_mfd_metrics(
	flow: np.ndarray,
	density: np.ndarray,
	segment_lengths: np.ndarray,
	coverage_mask: np.ndarray,
	region_ids: np.ndarray,
	eps: float = 1e-8,
) -> dict[str, float]:
	"""Calculate MFD metrics per region and their equal-weight regional average."""
	region_ids = np.asarray(region_ids)
	assert region_ids.shape == np.asarray(flow).shape, "region_ids must match segment-level flow shape."

	metrics_by_key = {}
	mean_values = {}
	for region_id in np.unique(region_ids):
		region_mask = region_ids == region_id
		region_metrics = calculate_mfd_metrics(
			flow=np.asarray(flow)[region_mask],
			density=np.asarray(density)[region_mask],
			segment_lengths=np.asarray(segment_lengths)[region_mask],
			coverage_mask=np.asarray(coverage_mask)[region_mask],
			eps=eps,
		)
		region_key = int(region_id)
		for key, value in region_metrics.items():
			if key == 'reward':
				continue
			metrics_by_key[f'region_{region_key}_{key}'] = value
			mean_values.setdefault(key, []).append(value)

	for key, values in mean_values.items():
		metrics_by_key[f'region_{key}'] = float(np.mean(values))
	metrics_by_key['region_mfd_reward'] = float(-metrics_by_key['region_mfd_error'])
	return metrics_by_key
