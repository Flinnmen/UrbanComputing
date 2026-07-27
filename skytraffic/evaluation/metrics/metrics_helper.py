import torch
import numpy as np
from typing import List, Tuple

from ...models.layers import GMMPredictionHead


def get_point_cdf(tensor: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """ tensor has shape (N, T, P), can be ground-truth or predicted most-likely values.
        xs has shape (X,), which is the points to evaluate the CDF.

        The CDF is a step function jumping at the value of tensor, with shape (N, T, P, X).
    """
    return (tensor.unsqueeze(-1) < xs).float()


def get_gmm_cdf(mixing: torch.Tensor, means: torch.Tensor, log_var: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    return GMMPredictionHead.get_mixture_cdf(mixing, means, log_var, xs)


def _evaluation_device(gpu: bool) -> torch.device:
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gmm_interval_coverage_and_width(
        mixing: torch.Tensor,
        means: torch.Tensor,
        log_var: torch.Tensor,
        gt: torch.Tensor,
        xs: torch.Tensor,
        conf: float,
        gpu: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate GMM confidence interval coverage and width."""
        device = _evaluation_device(gpu)
        xs = xs.to(device)
        mixing = mixing.to(device)
        means = means.to(device)
        log_var = log_var.to(device)
        gt = gt.to(device)

        lb, ub = GMMPredictionHead.get_confidence_interval(
            mixing, means, log_var, xs, conf=conf
        )
        within_ci = torch.any(
            torch.logical_and(
                lb <= gt.unsqueeze(-1),
                ub >= gt.unsqueeze(-1),
            ),
            dim=-1,
        )
        interval_width = torch.abs(ub - lb).sum(dim=-1)
        return within_ci.cpu(), interval_width.cpu()


def get_crps_gmm_vs_gt(mixing, means, log_var, xs, gt, gpu=True):
    """
    Compute CRPS between GMM prediction and ground truth values.

    Args:
        mixing: GMM mixing coefficients (N, T, P, K)
        means: GMM means (N, T, P, K)
        log_var: GMM log variances (N, T, P, K)
        xs: Points to evaluate density (X,)
        gt: Ground truth data (N, T, P)
        gpu: Whether to use GPU

    Returns:
        CRPS scores (N, T, P)
    """
    device = _evaluation_device(gpu)
    xs = xs.to(device)
    mixing = mixing.to(device)
    means = means.to(device)
    log_var = log_var.to(device)
    gt = gt.to(device)

    gmm_cdf = get_gmm_cdf(mixing, means, log_var, xs)
    gt_cdf = get_point_cdf(gt, xs)
    return torch.sum((gmm_cdf - gt_cdf) ** 2 * abs(xs[1] - xs[0]), dim=-1).cpu()


def get_gmm_crps_and_ci_scores(mixing, means, log_var, xs, gt, confs, gpu=True):
    """Compute CRPS and all confidence interval scores from one density grid."""
    device = _evaluation_device(gpu)
    xs = xs.to(device)
    mixing = mixing.to(device)
    means = means.to(device)
    log_var = log_var.to(device)
    gt = gt.to(device)

    dx = abs(xs[1] - xs[0])
    mixture_density = GMMPredictionHead.get_mixture_density(mixing, means, log_var, xs)

    gmm_cdf = torch.cumsum(mixture_density * dx, dim=-1)
    gmm_cdf = gmm_cdf / gmm_cdf[..., -1].unsqueeze(-1)
    gt_cdf = get_point_cdf(gt, xs)
    crps_scores = torch.sum((gmm_cdf - gt_cdf) ** 2 * dx, dim=-1)

    values, indexes = torch.sort(mixture_density, dim=-1, descending=True)
    prob_mass = (values * dx).cumsum(dim=-1)
    prob_mass = prob_mass / prob_mass[..., -1:]
    inv_index = torch.argsort(indexes, dim=-1)

    num_comp = mixing.shape[-1]
    index_range = torch.arange(xs.size(-1) + 1, device=xs.device)
    fill_index = torch.full_like(index_range, fill_value=xs.size(-1) - 1, device=xs.device)
    ci_cover, ci_width = {}, {}

    for conf in confs:
        in_interval = prob_mass <= conf
        x_is_in_interval = torch.gather(in_interval, dim=-1, index=inv_index)
        x_is_in_interval = torch.cat([
            torch.zeros_like(x_is_in_interval[..., :1]).bool(),
            x_is_in_interval,
            torch.zeros_like(x_is_in_interval[..., :1]).bool(),
        ], dim=-1)

        diff = torch.diff(x_is_in_interval.int(), dim=-1)
        lb_index = torch.sort(torch.where(diff == 1, index_range, fill_index))[0][..., :num_comp]
        ub_index = torch.sort(torch.where(diff == -1, index_range - 1, fill_index))[0][..., :num_comp]
        lb = xs[lb_index]
        ub = xs[ub_index]

        within_ci = torch.any(
            torch.logical_and(
                lb <= gt.unsqueeze(-1),
                ub >= gt.unsqueeze(-1),
            ),
            dim=-1,
        )
        ci_cover[conf] = within_ci.cpu()
        ci_width[conf] = torch.abs(ub - lb).sum(dim=-1).cpu()

    return crps_scores.cpu(), ci_cover, ci_width


def get_sample_crps_vs_gt(samples: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute CRPS for empirical predictive samples.

    Args:
        samples: Predictive samples with shape (..., K), where K is the sample axis.
        gt: Ground truth tensor with shape samples.shape[:-1].
    """
    from properscoring import crps_ensemble

    crps = crps_ensemble(
        gt.detach().cpu().numpy(),
        samples.detach().cpu().numpy(),
    )
    return torch.as_tensor(crps, dtype=torch.float32)


def sample_interval_coverage_and_width(
    samples: torch.Tensor,
    gt: torch.Tensor,
    conf: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate central confidence interval coverage and width from samples."""
    alpha = (1.0 - conf) / 2.0
    bounds = torch.quantile(samples.detach().cpu(), torch.tensor([alpha, 1.0 - alpha]), dim=-1)
    lb, ub = bounds[0], bounds[1]
    gt = gt.detach().cpu()
    return (lb <= gt) & (gt <= ub), ub - lb


def get_sample_crps_and_ci_scores(samples: torch.Tensor, gt: torch.Tensor, confs: List[float]):
    """Compute CRPS and central confidence interval scores from empirical samples."""
    crps_scores = get_sample_crps_vs_gt(samples, gt)
    ci_cover, ci_width = {}, {}

    for conf in confs:
        ci_cover[conf], ci_width[conf] = sample_interval_coverage_and_width(samples, gt, conf)

    return crps_scores, ci_cover, ci_width


def _quantile_level_tensor(quantile_levels, device: torch.device) -> torch.Tensor:
    levels = torch.as_tensor(quantile_levels, dtype=torch.float32, device=device)
    if torch.any(levels[1:] <= levels[:-1]):
        raise ValueError("quantile levels must be strictly increasing")
    return levels


def _quantile_integration_weights(quantile_levels: torch.Tensor) -> torch.Tensor:
    """Trapezoidal weights normalized over the provided quantile-level range."""
    if quantile_levels.numel() == 1:
        return torch.ones_like(quantile_levels)

    weights = torch.zeros_like(quantile_levels)
    weights[0] = (quantile_levels[1] - quantile_levels[0]) / 2
    weights[-1] = (quantile_levels[-1] - quantile_levels[-2]) / 2
    if quantile_levels.numel() > 2:
        weights[1:-1] = (quantile_levels[2:] - quantile_levels[:-2]) / 2
    return weights / weights.sum()


def get_quantile_mean(quantiles: torch.Tensor, quantile_levels) -> torch.Tensor:
    """Approximate the predictive mean by integrating the predicted quantile function."""
    levels = _quantile_level_tensor(quantile_levels, quantiles.device)
    weights = _quantile_integration_weights(levels)
    return (quantiles * weights.reshape(*([1] * (quantiles.ndim - 1)), -1)).sum(dim=-1)


def get_quantile_crps_vs_gt(
    quantiles: torch.Tensor,
    quantile_levels,
    gt: torch.Tensor,
) -> torch.Tensor:
    """Approximate CRPS using the quantile-score identity."""
    levels = _quantile_level_tensor(quantile_levels, quantiles.device)
    weights = _quantile_integration_weights(levels)
    errors = gt.to(quantiles.device).unsqueeze(-1) - quantiles
    pinball = torch.maximum(levels * errors, (levels - 1.0) * errors)
    return 2.0 * (pinball * weights.reshape(*([1] * (quantiles.ndim - 1)), -1)).sum(dim=-1)


def interpolate_quantile_values(
    quantiles: torch.Tensor,
    quantile_levels,
    target_levels,
) -> torch.Tensor:
    """Linearly interpolate predicted quantiles at target probability levels."""
    levels = _quantile_level_tensor(quantile_levels, quantiles.device)
    targets = torch.as_tensor(target_levels, dtype=torch.float32, device=quantiles.device)
    targets = targets.clamp(levels[0], levels[-1])

    hi = torch.searchsorted(levels, targets, right=False).clamp(1, levels.numel() - 1)
    lo = hi - 1
    lo_levels = levels[lo]
    hi_levels = levels[hi]
    weight = (targets - lo_levels) / (hi_levels - lo_levels)

    lo_values = torch.index_select(quantiles, dim=-1, index=lo)
    hi_values = torch.index_select(quantiles, dim=-1, index=hi)
    return lo_values + weight.reshape(*([1] * (quantiles.ndim - 1)), -1) * (hi_values - lo_values)


def quantile_interval_coverage_and_width(
    quantiles: torch.Tensor,
    quantile_levels,
    gt: torch.Tensor,
    conf: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate central confidence interval coverage and width from predicted quantiles."""
    alpha = (1.0 - conf) / 2.0
    bounds = interpolate_quantile_values(quantiles, quantile_levels, [alpha, 1.0 - alpha])
    lb, ub = bounds[..., 0], bounds[..., 1]
    gt = gt.to(quantiles.device)
    return ((lb <= gt) & (gt <= ub)).cpu(), (ub - lb).cpu()


def get_quantile_crps_and_ci_scores(
    quantiles: torch.Tensor,
    quantile_levels,
    gt: torch.Tensor,
    confs: List[float],
):
    """Compute quantile CRPS and confidence interval scores."""
    crps_scores = get_quantile_crps_vs_gt(quantiles, quantile_levels, gt).detach().cpu()
    ci_cover, ci_width = {}, {}

    for conf in confs:
        ci_cover[conf], ci_width[conf] = quantile_interval_coverage_and_width(
            quantiles,
            quantile_levels,
            gt,
            conf,
        )

    return crps_scores, ci_cover, ci_width


def ignore_score_when_gt_is(scores: torch.Tensor, gt: torch.Tensor, invalid_gt: float = float("nan"), ignore_value: float = float("nan")) -> torch.Tensor:
    """
    Ignore values in the scores tensor where the ground truth (gt) is NaN.

    Args:
        scores: Tensor of CRPS scores
        gt: Ground truth tensor
    """
    # when scores is boolean, setting ignore_value to NaN will make it True ...
    # so we need to convert it to float in that case
    if np.isnan(ignore_value) and not torch.is_floating_point(scores):
        scores = scores.float()

    if not np.isnan(invalid_gt):
        scores[gt == invalid_gt] = ignore_value

    # always ignore NaN values in gt
    scores[gt.isnan()] = ignore_value

    return scores
