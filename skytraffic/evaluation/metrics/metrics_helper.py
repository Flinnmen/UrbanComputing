import torch
import numpy as np
from typing import Tuple

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
