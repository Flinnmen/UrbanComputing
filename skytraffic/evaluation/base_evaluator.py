import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import common_metrics
from ..utils.io import make_dir_if_not_exist

logger = logging.getLogger("default")

class BaseEvaluator(ABC):
    """ Base class for all evaluators, with the `evaluate` method being an abstract method to be implemented in subclasses.
     
        We make a few assumptions:
        1. The model output and data batch are dictionaries of sequence names and data tensors of shape (N, T, P).
        2. The evaluator will collect the predictions and labels from the model output and data loader, and then evaluate the model.
        3. The evaluator will collect the scores as scalars or vectors, and save them to a JSON file.
        4. The evaluator can optionally visualize (and save) the evaluation results, this usually happens after training. 

        Note on tensor shapes:
            N: samples (batch size)
            T: time steps (input window or prediction horizon)
            P: spatial locations 
    """

    ignore_value: float
    mape_threshold: float

    def __init__(
        self,
        save_dir: str = None,
        collect_pred: List[str] = None,
        collect_data: List[str] = None,
    ) -> None:
        self.save_dir = save_dir
        make_dir_if_not_exist(self.save_dir)
        self.collect_pred = collect_pred
        self.collect_data = collect_data
        self.metrics_scalar: Dict[str, float] = dict()
        self.metrics_vector: Dict[str, List] = dict()

    def __call__(self, model: nn.Module, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        verbose = kwargs.pop('verbose', False)
        visualize = kwargs.pop('visualize', False)

        eval_res = self.evaluate(model, dataloader, verbose=verbose, visualize=visualize)

        if visualize:
            self.save_scores_to_json()

        return eval_res
    
    def collect_predictions(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        pred_seqs: List = None,
        data_seqs: List = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """run inference of the model on the data_loader concatenate all predictions and corresponding labels.
        
        The sequences in `data_seqs` will be collected from the data batch. 
        The sequences in `pred_seqs` will be collected from the model output. 
        Usually they are the same, but can be different when evaluation requires additional sequences.
        """
        model.eval()

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)

        for data_dict in data_loader:
            with torch.no_grad():
                pred_dict = model(data_dict)
            
            # collect predicted sequences corresponding labels
            for name in pred_seqs:
                all_preds[name].append(pred_dict[name])
            for name in data_seqs: 
                all_labels[name].append(data_dict[name])
                
        # this will actually modify all_preds and all_labels
        for res_collection in [all_preds, all_labels]:
            for key, value in res_collection.items():
                res_collection[key] = torch.cat(value, dim=0).detach().cpu()
        
        return all_preds, all_labels

    @abstractmethod
    def evaluate(self, model: nn.Module, dataloader: DataLoader, verbose: bool = False, visualize: bool = False) -> Dict[str, float]:
        pass

    def common_metrics_by_horizon(self, pred, label, verbose:bool=False) -> Dict[str, List[float]]:
        """ Both pred and label are tensors of shape (N, T, P), where T is the prediction horizon.
            This function then computes common error metrics (MAE, MAPE, RMSE) at each time step. 
        """
        eval_res_over_time = defaultdict(list)
        pred_steps = pred.shape[1]
        for i in range(pred_steps):  # number of predicted time step
            pred_i = pred[:, i, :]
            real_i = label[:, i, :]
            step_res = common_metrics(pred_i, real_i, self.ignore_value, self.mape_threshold)

            if verbose:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in step_res.items()])
                logger.info(f"Step {i} results: {metrics_str}")

            for k, v in step_res.items():
                eval_res_over_time[k].append(v)

        return eval_res_over_time
    

    def average_metrics(self, eval_res_by_horizon, verbose: bool = False) -> Dict[str, float]:
        """ Compute average metrics as a simple arithmetic average of the per-horizon results.
            This is slightly different from computing the average metrics directly from all predictions and labels, 
            e.g., mae = (pred - label).abs().mean(), because the mask may be different at different horizons.
        """
        avg_eval_res = {k:sum(v)/len(v) for k, v in eval_res_by_horizon.items()}
        if verbose:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_eval_res.items()])
            logger.info(f"Average results: {metrics_str}")
        return avg_eval_res
    

    def save_scores_to_json(self, file_name: str = "final_evaluation_scores.json"):
        """
        Save the scores to a JSON file.
        The scores are saved in a dictionary with keys being the score types and values being the scores.
        """
        scalar_res = {k:float(v) for k, v in self.metrics_scalar.items()}
        vector_res = {k:v for k, v in self.metrics_vector.items() if isinstance(v, list)}
        res_to_save = {
            "average": scalar_res,
            "horizon": vector_res
        }

        save_path = f"{self.save_dir}/{file_name}"
        with open(save_path, 'w') as f:
            json.dump(res_to_save, f, indent=4)
        logger.info(f"Saved scores to {save_path}")


    def analyze_scores(self, scores: torch.Tensor, note: str, verbose: bool = False):
        """
        Analyze and store evaluation scores.
        
        Args:
            scores: Score tensor with shape (N, T, P)
            note: Key to use when storing scores
            verbose: Whether to print summary
        """
        if note is None or note == "":
            logger.error(f"The note cannot be {note}, please use a different one.")
            return
        
        # Save average score at each time step
        self.metrics_vector[note] = torch.nanmean(scores, dim=(0, 2)).cpu().numpy().tolist()
        # Save overall mean score
        self.metrics_scalar[note] = torch.nanmean(scores).item()
        
        if verbose:
            logger.info(f"Overall average {note}: {self.metrics_scalar[note]:.4f}")

    def valid_target_mask(self, target: torch.Tensor) -> torch.Tensor:
        valid = ~target.isnan()
        if not np.isnan(self.ignore_value):
            valid = valid & (target != self.ignore_value)
        return valid

    def add_scores_to_accumulator(
        self,
        score_sums_per_horizon: Dict[str, torch.Tensor],
        score_counts_per_horizon: Dict[str, torch.Tensor],
        name: str,
        scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        scores = scores.detach().cpu().to(torch.float64)
        valid = valid.detach().cpu()
        if name not in score_sums_per_horizon:
            score_sums_per_horizon[name] = torch.zeros(scores.shape[1], dtype=torch.float64)
            score_counts_per_horizon[name] = torch.zeros(scores.shape[1], dtype=torch.float64)

        score_sums_per_horizon[name] += torch.where(valid, scores, torch.zeros_like(scores)).sum(dim=(0, 2))
        score_counts_per_horizon[name] += valid.sum(dim=(0, 2)).to(torch.float64)

    def weighted_mean_by_horizon(
        self,
        score_sums_per_horizon: Dict[str, torch.Tensor],
        score_counts_per_horizon: Dict[str, torch.Tensor],
        name: str,
        empty_value: float = float("nan"),
    ) -> List[float]:
        values = torch.full_like(score_sums_per_horizon[name], empty_value)
        valid = score_counts_per_horizon[name] > 0
        values[valid] = score_sums_per_horizon[name][valid] / score_counts_per_horizon[name][valid]
        return values.cpu().numpy().tolist()

    def store_weighted_score(
        self,
        score_sums_per_horizon: Dict[str, torch.Tensor],
        score_counts_per_horizon: Dict[str, torch.Tensor],
        name: str,
        verbose: bool = False,
    ) -> None:
        self.metrics_vector[name] = self.weighted_mean_by_horizon(
            score_sums_per_horizon,
            score_counts_per_horizon,
            name,
        )
        if score_counts_per_horizon[name].sum() > 0:
            self.metrics_scalar[name] = (
                score_sums_per_horizon[name].sum() / score_counts_per_horizon[name].sum()
            ).item()
        else:
            self.metrics_scalar[name] = float("nan")

        if verbose:
            logger.info(f"Overall average {name}: {self.metrics_scalar[name]:.4f}")

    def add_common_prediction_scores(
        self,
        score_sums_per_horizon: Dict[str, torch.Tensor],
        score_counts_per_horizon: Dict[str, torch.Tensor],
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        valid = valid.detach().cpu()

        if torch.isnan(pred).any():
            logger.warning(f"number of nan in model predictions: {torch.sum(torch.isnan(pred)).item()}")

        error = pred - target
        self.add_scores_to_accumulator(score_sums_per_horizon, score_counts_per_horizon, "mae", error.abs(), valid)
        self.add_scores_to_accumulator(score_sums_per_horizon, score_counts_per_horizon, "mse", error.pow(2), valid)

        mape_valid = valid
        if self.mape_threshold is not None:
            mape_valid = mape_valid & (target.abs() > self.mape_threshold)
        mape = torch.zeros_like(target)
        mape[mape_valid] = (error[mape_valid] / target[mape_valid]).abs()
        self.add_scores_to_accumulator(score_sums_per_horizon, score_counts_per_horizon, "mape", mape, mape_valid)

    def store_common_prediction_scores(
        self,
        score_sums_per_horizon: Dict[str, torch.Tensor],
        score_counts_per_horizon: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> None:
        self.metrics_vector["mae"] = self.weighted_mean_by_horizon(
            score_sums_per_horizon,
            score_counts_per_horizon,
            "mae",
            empty_value=0.0,
        )
        self.metrics_vector["mape"] = self.weighted_mean_by_horizon(
            score_sums_per_horizon,
            score_counts_per_horizon,
            "mape",
            empty_value=0.0,
        )
        rmse = torch.tensor(
            self.weighted_mean_by_horizon(score_sums_per_horizon, score_counts_per_horizon, "mse")
        )
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

    def compute_interval_calibration_metrics(self, confs: List[float], verbose: bool = False):
        CCE_conf_horizon, AW_conf_horizon = [], []

        for conf in confs:
            CCE_conf_horizon.append([abs(x - conf) for x in self.metrics_vector[f"CI_COVER_{conf}"]])
            AW_conf_horizon.append(self.metrics_vector[f"CI_WIDTH_{conf}"])

        mCCE_horizon = np.stack(CCE_conf_horizon, axis=-1).mean(axis=-1)
        mAW_horizon = np.stack(AW_conf_horizon, axis=-1).mean(axis=-1)

        self.metrics_scalar["mCCE"] = mCCE_horizon.mean().item()
        self.metrics_scalar["mAW"] = mAW_horizon.mean().item()
        self.metrics_vector["mCCE"] = mCCE_horizon.tolist()
        self.metrics_vector["mAW"] = mAW_horizon.tolist()

        if verbose:
            logger.info(f"Mean Confidence Calibration Error (mCCE): {self.metrics_scalar['mCCE']:.4f}")
            logger.info(f"Mean Average Width (mAW): {self.metrics_scalar['mAW']:.4f}")
