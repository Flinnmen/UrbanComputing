import copy
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from typing import Dict

from .metr_evaluation import MetrEvaluator
from .metrics import (
    gmm_interval_coverage_and_width,
    ignore_score_when_gt_is,
    get_crps_gmm_vs_gt,
    get_gmm_crps_and_ci_scores,
)

logger = logging.getLogger("default")

class MetrGMMEvaluator(MetrEvaluator):
    """
    Evaluator for Gaussian Mixture Model predictions on METR dataset.
    Extends MetrEvaluator with uncertainty evaluation capabilities.
    """
    # confidence levels to evaluate
    eval_confs = np.round(np.arange(0.5, 1.0, 0.05), 2).tolist()  

    def __init__(self, 
                 save_dir: str = None, 
                 collect_pred=["pred", "mixing", "means", "log_var"], 
                 collect_data=["target"],
                 data_min: float = 0.0, 
                 data_max: float = 70.0,
                 gpu: bool = True,
                 ci_pts: int = 500):
        super().__init__(save_dir, collect_pred, collect_data)
        self.data_min = data_min # minimum data value (traffic data is typically non-negative)
        self.data_max = data_max # maximum data value (adjustable based on dataset)
        self.gpu = gpu # whether to use GPU acceleration for evaluation
        self.ci_pts = ci_pts # number of points to evaluate GMM density for confidence intervals

    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False, visualize: bool = False) -> Dict[str, float]:
        """
        Evaluate GMM model with both deterministic and probabilistic metrics.
        
        Args:
            model: The GMM model to evaluate
            dataloader: Data loader for evaluation
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        xs = torch.linspace(self.data_min, self.data_max, self.ci_pts)

        score_sums_per_horizon = {}
        score_counts_per_horizon = {}

        def add_scores(name: str, scores: torch.Tensor, valid: torch.Tensor) -> None:
            scores = scores.detach().cpu().to(torch.float64)
            valid = valid.detach().cpu()
            if name not in score_sums_per_horizon:
                score_sums_per_horizon[name] = torch.zeros(scores.shape[1], dtype=torch.float64)
                score_counts_per_horizon[name] = torch.zeros(scores.shape[1], dtype=torch.float64)

            score_sums_per_horizon[name] += torch.where(valid, scores, torch.zeros_like(scores)).sum(dim=(0, 2))
            score_counts_per_horizon[name] += valid.sum(dim=(0, 2)).to(torch.float64)

        def mean_by_horizon(name: str, empty_value: float = float("nan")) -> list:
            values = torch.full_like(score_sums_per_horizon[name], empty_value)
            valid = score_counts_per_horizon[name] > 0
            values[valid] = score_sums_per_horizon[name][valid] / score_counts_per_horizon[name][valid]
            return values.cpu().numpy().tolist()

        def store_weighted_mean(name: str, verbose: bool = False) -> None:
            self.metrics_vector[name] = mean_by_horizon(name)
            if score_counts_per_horizon[name].sum() > 0:
                self.metrics_scalar[name] = (score_sums_per_horizon[name].sum() / score_counts_per_horizon[name].sum()).item()
            else:
                self.metrics_scalar[name] = float("nan")

            if verbose:
                logger.info(f"Overall average {name}: {self.metrics_scalar[name]:.4f}")

        for data_dict in dataloader:
            with torch.no_grad():
                pred_dict = model(data_dict)

            target = data_dict["target"]
            valid = self.valid_target_mask(target)
            pred = pred_dict["pred"].detach().cpu()
            target_cpu = target.detach().cpu()
            valid_cpu = valid.detach().cpu()

            if torch.isnan(pred).any():
                logger.warning(f"number of nan in model predictions: {torch.sum(torch.isnan(pred)).item()}")

            error = pred - target_cpu
            add_scores("mae", error.abs(), valid_cpu)
            add_scores("mse", error.pow(2), valid_cpu)

            mape_valid = valid_cpu
            if self.mape_threshold is not None:
                mape_valid = mape_valid & (target_cpu.abs() > self.mape_threshold)
            mape = torch.zeros_like(target_cpu)
            mape[mape_valid] = (error[mape_valid] / target_cpu[mape_valid]).abs()
            add_scores("mape", mape, mape_valid)

            crps_scores, ci_cover, ci_width = get_gmm_crps_and_ci_scores(
                mixing=pred_dict["mixing"],
                means=pred_dict["means"],
                log_var=pred_dict["log_var"],
                xs=xs,
                gt=target,
                confs=self.eval_confs,
                gpu=self.gpu,
            )
            add_scores("CRPS_GMM_GT", crps_scores, valid_cpu)

            for conf in self.eval_confs:
                add_scores(f"CI_COVER_{conf}", ci_cover[conf].float(), valid_cpu)
                add_scores(f"CI_WIDTH_{conf}", ci_width[conf], valid_cpu)

        self.metrics_vector["mae"] = mean_by_horizon("mae", empty_value=0.0)
        self.metrics_vector["mape"] = mean_by_horizon("mape", empty_value=0.0)
        rmse = torch.tensor(mean_by_horizon("mse"))
        self.metrics_vector["rmse"] = torch.sqrt(rmse).numpy().tolist()
        self.metrics_scalar.update(
            self.average_metrics(
                {
                    "mae": self.metrics_vector["mae"],
                    "mape": self.metrics_vector["mape"],
                    "rmse": self.metrics_vector["rmse"],
                },
                verbose=verbose,
            )
        )

        store_weighted_mean("CRPS_GMM_GT", verbose=verbose)

        for conf in self.eval_confs:
            store_weighted_mean(f"CI_COVER_{conf}", verbose=False)
            store_weighted_mean(f"CI_WIDTH_{conf}", verbose=False)
        self.compute_calibration_metrics(verbose=verbose)

        return copy.deepcopy(self.metrics_scalar)

    def valid_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        valid = ~target.isnan()
        if not np.isnan(self.ignore_value):
            valid = valid & (target != self.ignore_value)
        return valid

    def evaluate_crps(self, all_preds: Dict[str, torch.Tensor], all_data: Dict[str, torch.Tensor], verbose: bool = False):
        """
        Evaluate CRPS (Continuous Ranked Probability Score) between GMM predictions and ground truth.
        
        Args:
            all_preds: Dictionary containing model predictions
            all_data: Dictionary containing ground truth data
            verbose: Whether to print results
        """
        # Define evaluation points
        xs = torch.linspace(self.data_min, self.data_max, self.ci_pts)
        
        # Evaluate CRPS between GMM and ground truth
        crps_scores = get_crps_gmm_vs_gt(
            mixing=all_preds["mixing"],
            means=all_preds["means"],
            log_var=all_preds["log_var"],
            xs=xs,
            gt=all_data["target"],
            gpu=self.gpu
        )
        
        # Ignore scores where ground truth is NaN
        crps_scores = ignore_score_when_gt_is(crps_scores, all_data["target"], invalid_gt=self.ignore_value)
        
        # Analyze and store scores
        self.analyze_scores(crps_scores, note="CRPS_GMM_GT", verbose=verbose)

    def evaluate_confidence_interval(self, all_preds: Dict[str, torch.Tensor], all_data: Dict[str, torch.Tensor], verbose: bool = False):
        """
        Evaluate confidence interval coverage and width for different confidence levels.
        
        Args:
            all_preds: Dictionary containing model predictions
            all_data: Dictionary containing ground truth data
            verbose: Whether to print results
        """
        xs = torch.linspace(self.data_min, self.data_max, self.ci_pts)
        
        for conf in self.eval_confs:
            within_ci, interval_width = gmm_interval_coverage_and_width(
                mixing=all_preds["mixing"],
                means=all_preds["means"],
                log_var=all_preds["log_var"],
                gt=all_data["target"],
                xs=xs,
                conf=conf,
                gpu=self.gpu
            )
            
            # Ignore scores where ground truth is NaN
            within_ci = ignore_score_when_gt_is(within_ci, all_data["target"], invalid_gt=self.ignore_value)
            interval_width = ignore_score_when_gt_is(interval_width, all_data["target"], invalid_gt=self.ignore_value)
            
            # Store individual confidence level results
            self.analyze_scores(within_ci.float(), note=f"CI_COVER_{conf}", verbose=False)
            self.analyze_scores(interval_width, note=f"CI_WIDTH_{conf}", verbose=False)
        
        # Compute overall calibration metrics
        self.compute_calibration_metrics(verbose=verbose)

    def compute_calibration_metrics(self, verbose: bool = False):
        """
        Compute overall calibration metrics from individual confidence level results.
        
        Args:
            verbose: Whether to print results
        """
        CCE_conf_horizon, AW_conf_horizon = [], []
        
        for conf in self.eval_confs:
            # Confidence Calibration Error: difference between confidence level and actual coverage
            CCE_conf_horizon.append([abs(x - conf) for x in self.metrics_vector[f'CI_COVER_{conf}']])
            AW_conf_horizon.append(self.metrics_vector[f'CI_WIDTH_{conf}'])
        
        # Average over confidence levels
        mCCE_horizon = np.stack(CCE_conf_horizon, axis=-1).mean(axis=-1)
        mAW_horizon = np.stack(AW_conf_horizon, axis=-1).mean(axis=-1)
        
        # Store overall metrics
        self.metrics_scalar['mCCE'] = mCCE_horizon.mean().item()
        self.metrics_scalar['mAW'] = mAW_horizon.mean().item()
        self.metrics_vector['mCCE'] = mCCE_horizon.tolist()
        self.metrics_vector['mAW'] = mAW_horizon.tolist()
        
        if verbose:
            logger.info(f"Mean Confidence Calibration Error (mCCE): {self.metrics_scalar['mCCE']:.4f}")
            logger.info(f"Mean Average Width (mAW): {self.metrics_scalar['mAW']:.4f}")
