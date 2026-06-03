import copy
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .metr_evaluation import MetrEvaluator
from .metrics import (
    get_quantile_crps_and_ci_scores,
    get_sample_crps_and_ci_scores,
)

logger = logging.getLogger("default")


class MetrMCDropoutEvaluator(MetrEvaluator):
    """Evaluate deterministic METR models as sample distributions via MC dropout."""

    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

    def __init__(
        self,
        save_dir: str = None,
        collect_pred: List[str] = None,
        collect_data: List[str] = None,
        num_samples: int = 30,
    ) -> None:
        super().__init__(save_dir, collect_pred or ["pred"], collect_data or ["target"])
        self.num_samples = num_samples

    def _stochastic_modules(self, model: nn.Module) -> List[nn.Module]:
        network = getattr(model, "network", model)
        if hasattr(network, "gconv") and len(network.gconv) > 0:
            return list(network.gconv)

        dropout_modules = [module for module in network.modules() if isinstance(module, nn.Dropout)]
        if dropout_modules:
            return dropout_modules

        raise ValueError("MC dropout evaluation requires at least one dropout-bearing module")

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        verbose: bool = False,
        visualize: bool = False,
    ) -> Dict[str, float]:
        model.eval()
        stochastic_modules = self._stochastic_modules(model)
        score_sums_per_horizon, score_counts_per_horizon = {}, {}

        for data_dict in dataloader:
            samples = []
            for _ in range(self.num_samples):
                model.eval()
                for module in stochastic_modules:
                    module.train()
                with torch.no_grad():
                    samples.append(model(data_dict)["pred"])

            model.eval()
            samples = torch.stack(samples, dim=-1).detach().cpu()
            pred = samples.mean(dim=-1)
            target = data_dict["target"].detach().cpu()
            valid = self.valid_target_mask(target)

            self.add_common_prediction_scores(
                score_sums_per_horizon,
                score_counts_per_horizon,
                pred,
                target,
                valid,
            )
            crps_scores, ci_cover, ci_width = get_sample_crps_and_ci_scores(
                samples,
                target,
                self.eval_confs,
            )
            self.add_scores_to_accumulator(
                score_sums_per_horizon,
                score_counts_per_horizon,
                "CRPS_SAMPLE_GT",
                crps_scores,
                valid,
            )

            for conf in self.eval_confs:
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_COVER_{conf}",
                    ci_cover[conf].float(),
                    valid,
                )
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_WIDTH_{conf}",
                    ci_width[conf],
                    valid,
                )

        self.store_common_prediction_scores(score_sums_per_horizon, score_counts_per_horizon, verbose=verbose)
        self.store_weighted_score(
            score_sums_per_horizon,
            score_counts_per_horizon,
            "CRPS_SAMPLE_GT",
            verbose=verbose,
        )
        for conf in self.eval_confs:
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_COVER_{conf}")
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_WIDTH_{conf}")
        self.compute_interval_calibration_metrics(self.eval_confs, verbose=verbose)

        return copy.deepcopy(self.metrics_scalar)


class MetrQuantileEvaluator(MetrEvaluator):
    """Evaluate METR quantile predictions."""

    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

    def __init__(
        self,
        save_dir: str = None,
        collect_pred: List[str] = None,
        collect_data: List[str] = None,
        quantile_levels: List[float] = None,
    ) -> None:
        super().__init__(save_dir, collect_pred or ["pred", "quantiles"], collect_data or ["target"])
        self.quantile_levels = quantile_levels

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        verbose: bool = False,
        visualize: bool = False,
    ) -> Dict[str, float]:
        model.eval()
        score_sums_per_horizon, score_counts_per_horizon = {}, {}

        for data_dict in dataloader:
            with torch.no_grad():
                pred_dict = model(data_dict)

            target = data_dict["target"].detach().cpu()
            valid = self.valid_target_mask(target)
            pred = pred_dict["pred"].detach().cpu()
            quantiles = pred_dict["quantiles"].detach().cpu()
            quantile_levels = self.quantile_levels or pred_dict.get("quantile_levels")
            if quantile_levels is None:
                raise ValueError("quantile_levels must be configured or returned by the model")

            self.add_common_prediction_scores(
                score_sums_per_horizon,
                score_counts_per_horizon,
                pred,
                target,
                valid,
            )
            crps_scores, ci_cover, ci_width = get_quantile_crps_and_ci_scores(
                quantiles,
                quantile_levels,
                target,
                self.eval_confs,
            )
            self.add_scores_to_accumulator(
                score_sums_per_horizon,
                score_counts_per_horizon,
                "CRPS_QUANTILE_GT",
                crps_scores,
                valid,
            )

            for conf in self.eval_confs:
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_COVER_{conf}",
                    ci_cover[conf].float(),
                    valid,
                )
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_WIDTH_{conf}",
                    ci_width[conf],
                    valid,
                )

        self.store_common_prediction_scores(score_sums_per_horizon, score_counts_per_horizon, verbose=verbose)
        self.store_weighted_score(
            score_sums_per_horizon,
            score_counts_per_horizon,
            "CRPS_QUANTILE_GT",
            verbose=verbose,
        )
        for conf in self.eval_confs:
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_COVER_{conf}")
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_WIDTH_{conf}")
        self.compute_interval_calibration_metrics(self.eval_confs, verbose=verbose)

        return copy.deepcopy(self.metrics_scalar)


class MetrConformalEvaluator(MetrEvaluator):
    """Evaluate deterministic METR predictions with split-conformal residual intervals."""

    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()

    def __init__(
        self,
        save_dir: str = None,
        collect_pred: List[str] = None,
        collect_data: List[str] = None,
        calibration_dataset: Dataset = None,
        calibration_dataloader: DataLoader = None,
        calibration_batch_size: int = 32,
        max_residual_samples: int = 128,
    ) -> None:
        super().__init__(save_dir, collect_pred or ["pred"], collect_data or ["target"])
        if calibration_dataloader is None and calibration_dataset is not None:
            calibration_dataloader = DataLoader(
                calibration_dataset,
                batch_size=calibration_batch_size,
                shuffle=False,
                collate_fn=calibration_dataset.collate_fn,
            )
        self.calibration_dataloader = calibration_dataloader
        self.max_residual_samples = max_residual_samples

    def _calibrate(self, model: nn.Module):
        if self.calibration_dataloader is None:
            raise ValueError("MetrConformalEvaluator requires a calibration dataloader or dataset")

        model.eval()
        residuals_by_horizon = None

        for data_dict in self.calibration_dataloader:
            with torch.no_grad():
                pred = model(data_dict)["pred"].detach().cpu()
            target = data_dict["target"].detach().cpu()
            valid = self.valid_target_mask(target)
            residual = target - pred

            if residuals_by_horizon is None:
                residuals_by_horizon = [[] for _ in range(residual.shape[1])]

            for horizon in range(residual.shape[1]):
                residuals_by_horizon[horizon].append(residual[:, horizon, :][valid[:, horizon, :]])

        residuals_by_horizon = [torch.cat(values) for values in residuals_by_horizon]
        abs_quantiles = {conf: [] for conf in self.eval_confs}
        signed_samples = []
        pairwise_terms = []

        for residuals in residuals_by_horizon:
            residuals = residuals[~residuals.isnan()]
            abs_residuals = residuals.abs()
            sample_count = min(self.max_residual_samples, residuals.numel())
            sample_idx = torch.linspace(0, residuals.numel() - 1, steps=sample_count).round().long()
            sampled_residuals = residuals[sample_idx].to(torch.float32)
            signed_samples.append(sampled_residuals)
            pairwise_terms.append(
                (sampled_residuals[:, None] - sampled_residuals[None, :]).abs().mean()
            )

            for conf in self.eval_confs:
                q_level = min(1.0, np.ceil((abs_residuals.numel() + 1) * conf) / abs_residuals.numel())
                abs_quantiles[conf].append(torch.quantile(abs_residuals, q_level).item())

        abs_quantiles = {
            conf: torch.as_tensor(values, dtype=torch.float32)
            for conf, values in abs_quantiles.items()
        }
        return torch.stack(signed_samples), torch.stack(pairwise_terms), abs_quantiles

    def _conformal_crps(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        signed_samples: torch.Tensor,
        pairwise_terms: torch.Tensor,
    ) -> torch.Tensor:
        device = pred.device
        residual_samples = signed_samples.to(device)
        pairwise_terms = pairwise_terms.to(device)
        errors = target.to(device) - pred
        mean_abs_error = (
            errors.unsqueeze(-1)
            - residual_samples.reshape(1, residual_samples.shape[0], 1, residual_samples.shape[1])
        ).abs().mean(dim=-1)
        return (mean_abs_error - 0.5 * pairwise_terms.reshape(1, -1, 1)).clamp_min(0.0).cpu()

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        verbose: bool = False,
        visualize: bool = False,
    ) -> Dict[str, float]:
        model.eval()
        signed_samples, pairwise_terms, abs_quantiles = self._calibrate(model)
        score_sums_per_horizon, score_counts_per_horizon = {}, {}

        for data_dict in dataloader:
            with torch.no_grad():
                pred_dict = model(data_dict)

            target = data_dict["target"].detach().cpu()
            valid = self.valid_target_mask(target)
            pred_device = pred_dict["pred"].detach()
            pred = pred_device.cpu()

            self.add_common_prediction_scores(
                score_sums_per_horizon,
                score_counts_per_horizon,
                pred,
                target,
                valid,
            )
            crps_scores = self._conformal_crps(
                pred_device,
                data_dict["target"].to(pred_device.device),
                signed_samples,
                pairwise_terms,
            )
            self.add_scores_to_accumulator(
                score_sums_per_horizon,
                score_counts_per_horizon,
                "CRPS_CONFORMAL_GT",
                crps_scores,
                valid,
            )

            for conf in self.eval_confs:
                half_width = abs_quantiles[conf].reshape(1, -1, 1)
                covered = (pred - half_width <= target) & (target <= pred + half_width)
                width = torch.full_like(target, 2.0) * half_width
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_COVER_{conf}",
                    covered.float(),
                    valid,
                )
                self.add_scores_to_accumulator(
                    score_sums_per_horizon,
                    score_counts_per_horizon,
                    f"CI_WIDTH_{conf}",
                    width,
                    valid,
                )

        self.store_common_prediction_scores(score_sums_per_horizon, score_counts_per_horizon, verbose=verbose)
        self.store_weighted_score(
            score_sums_per_horizon,
            score_counts_per_horizon,
            "CRPS_CONFORMAL_GT",
            verbose=verbose,
        )
        for conf in self.eval_confs:
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_COVER_{conf}")
            self.store_weighted_score(score_sums_per_horizon, score_counts_per_horizon, f"CI_WIDTH_{conf}")
        self.compute_interval_calibration_metrics(self.eval_confs, verbose=verbose)

        return copy.deepcopy(self.metrics_scalar)
