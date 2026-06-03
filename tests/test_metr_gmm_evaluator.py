import copy
import json
import unittest
import torch
import numpy as np
import tempfile
import os
import torch.nn as nn

from typing import Dict
from skytraffic.data.datasets import MetrDataset
from skytraffic.evaluation.metr_evaluation import MetrEvaluator
from skytraffic.evaluation.metr_gmm_evaluation import MetrGMMEvaluator
from skytraffic.evaluation.metrics import (
    gmm_interval_coverage_and_width,
    get_crps_gmm_vs_gt,
    ignore_score_when_gt_is,
)
from torch.utils.data import Dataset, DataLoader, Subset


class LegacyTwoPassMetrGMMEvaluator(MetrEvaluator):
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
        # First get basic deterministic metrics from parent class
        _ = super().evaluate(model, dataloader, verbose=verbose, visualize=visualize)

        # Collect predictions and data for GMM evaluation
        all_preds, all_data = self.collect_predictions(
            model, dataloader,
            pred_seqs=self.collect_pred,
            data_seqs=self.collect_data
        )

        self.evaluate_crps(all_preds, all_data, verbose=verbose)

        self.evaluate_confidence_interval(all_preds, all_data, verbose=verbose)

        # Update results with GMM metrics
        return copy.deepcopy(self.metrics_scalar)

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
        crps_scores = ignore_score_when_gt_is(crps_scores, all_data["target"], invalid_gt=0.0)

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
            within_ci = ignore_score_when_gt_is(within_ci, all_data["target"], invalid_gt=0.0)
            interval_width = ignore_score_when_gt_is(interval_width, all_data["target"], invalid_gt=0.0)

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


class DeterministicMetrGMMModel(nn.Module):
    """Small deterministic GMM head for comparing evaluator implementations on real METR inputs."""

    def __init__(self, pred_steps=12, num_mixtures=3):
        super().__init__()
        self.pred_steps = pred_steps
        self.num_mixtures = num_mixtures

    def forward(self, x):
        source_speed = x["source"][:, -self.pred_steps:, :, 0]
        base = source_speed.clamp(0.0, 70.0)
        offsets = torch.linspace(
            -2.0,
            2.0,
            self.num_mixtures,
            device=base.device,
            dtype=base.dtype,
        ).reshape(1, 1, 1, self.num_mixtures)
        mixing = torch.full(
            (*base.shape, self.num_mixtures),
            fill_value=1.0 / self.num_mixtures,
            device=base.device,
            dtype=base.dtype,
        )
        means = (base.unsqueeze(-1) + offsets).clamp(0.0, 70.0)
        log_var = torch.full_like(means, fill_value=1.0)
        pred = (mixing * means).sum(dim=-1)

        return {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var,
        }


class FixedDataset(Dataset):
    """Dataset with stable tensors so legacy and streaming evaluators see identical batches."""

    def __init__(self, num_samples=4, num_nodes=4, input_steps=3, pred_steps=2):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        source_values = torch.arange(
            num_samples * input_steps * num_nodes * 2,
            dtype=torch.float32,
        )
        target_values = torch.arange(
            num_samples * pred_steps * num_nodes,
            dtype=torch.float32,
        )
        self.sources = source_values.reshape(num_samples, input_steps, num_nodes, 2) / 10.0 + 1.0
        self.targets = target_values.reshape(num_samples, pred_steps, num_nodes) / 10.0 + 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "source": self.sources[idx],
            "target": self.targets[idx],
        }

    def collate_fn(self, batch):
        return {
            "source": torch.stack([item["source"] for item in batch], dim=0),
            "target": torch.stack([item["target"] for item in batch], dim=0),
        }


class InvalidTargetDataset(Dataset):
    """Single-batch fixture with both METR invalid zero and NaN targets."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "source": torch.zeros(2, 4, 2),
            "target": torch.tensor([[0.0, float("nan"), 3.0, 5.0]]),
        }

    def collate_fn(self, batch):
        return {
            "source": torch.stack([item["source"] for item in batch], dim=0),
            "target": torch.stack([item["target"] for item in batch], dim=0),
        }


class FixedPredictionGMMModel(nn.Module):
    def forward(self, x):
        target = x["target"]
        pred = torch.tensor(
            [[[10.0, 10.0, 1.0, 7.0]]],
            device=target.device,
            dtype=target.dtype,
        ).expand_as(target).clone()
        mixing = torch.ones((*target.shape, 1), device=target.device, dtype=target.dtype)
        means = pred.unsqueeze(-1)
        log_var = torch.zeros_like(means)
        return {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var,
        }


def assert_metric_dict_allclose(testcase, actual, expected):
    testcase.assertEqual(set(actual.keys()), set(expected.keys()))
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-6, equal_nan=True)


class DummyDataset(Dataset):
    """Simple dataset that generates random data"""

    def __init__(self, num_samples=64, num_nodes=207, input_steps=12, pred_steps=12):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.pred_steps = pred_steps

        # Create dummy metadata once
        self.metadata = {
            'adjacency': [torch.randn(num_nodes, num_nodes)],
            'mean': torch.tensor(0.0),
            'std': torch.tensor(1.0),
            'data_dim': 0,
            'geo_loc': np.random.randn(num_nodes, 2),
            'invalid_value': 0.0
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "source": torch.randn(self.input_steps, self.num_nodes, 2),
            "target": torch.randn(self.pred_steps, self.num_nodes),
        }

    def collate_fn(self, batch):
        """ assume the input is a list of (x, y), pack the x's and y's into two tensors
        """
        xs = [torch.as_tensor(xy["source"]).unsqueeze(0) for xy in batch]
        ys = [torch.as_tensor(xy["target"]).unsqueeze(0) for xy in batch]

        xs, ys = torch.cat(xs, dim=0), torch.cat(ys, dim=0)

        return {"source": xs.contiguous(), "target": ys.contiguous()}

def dummy_dataloader(batch_size=16, num_nodes=207, input_steps=12, pred_steps=12, num_samples=64):
    """Create a DataLoader with dummy data"""
    dataset = DummyDataset(
        num_samples=num_samples,
        num_nodes=num_nodes,
        input_steps=input_steps,
        pred_steps=pred_steps
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

class DummyModel(nn.Module):
    def __init__(self, num_nodes=207, pred_steps=12, num_mixtures=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_steps = pred_steps
        self.num_mixtures = num_mixtures

    def forward(self, x):
        """Simple function that creates dummy GMM outputs"""
        # Generate dummy GMM outputs
        batch_size = x['source'].size(0)
        pred = torch.randn(batch_size, self.pred_steps, self.num_nodes)
        mixing = torch.softmax(torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures), dim=-1)
        means = torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures)
        log_var = torch.randn(batch_size, self.pred_steps, self.num_nodes, self.num_mixtures)

        return {
            "pred": pred,
            "mixing": mixing,
            "means": means,
            "log_var": log_var
        }


class CountingDummyModel(DummyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return super().forward(x)


class TestMetrGMMEvaluator(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.num_nodes = 207
        self.input_steps = 12
        self.pred_steps = 12
        self.batch_size = 16

        # Create simple dataloader
        self.dataloader = dummy_dataloader(
            batch_size=self.batch_size,
            num_nodes=self.num_nodes,
            input_steps=self.input_steps,
            pred_steps=self.pred_steps
        )

        # Model parameters for creating dummy outputs
        self.num_mixtures = 5

        # Create simple mock model
        self.model = DummyModel(num_nodes=self.num_nodes, pred_steps=self.pred_steps, num_mixtures=self.num_mixtures)

        # Create temporary directory for evaluator
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def test_collect_predictions(self):
        """Test that the evaluator can collect predictions correctly"""
        evaluator = MetrGMMEvaluator(save_dir=self.temp_dir)

        all_preds, all_data = evaluator.collect_predictions(
            self.model,
            self.dataloader,
            pred_seqs=["pred", "mixing", "means", "log_var"],
            data_seqs=["target"]
        )

        # Check that predictions were collected
        self.assertIn("pred", all_preds)
        self.assertIn("mixing", all_preds)
        self.assertIn("means", all_preds)
        self.assertIn("log_var", all_preds)
        self.assertIn("target", all_data)

        # Check shapes are correct
        expected_samples = 4 * self.batch_size  # 4 batches * batch_size
        self.assertEqual(all_preds["pred"].shape[0], expected_samples)
        self.assertEqual(all_data["target"].shape[0], expected_samples)

    def test_evaluate_method(self):
        """Test the main evaluate method"""
        evaluator = MetrGMMEvaluator(save_dir=self.temp_dir)

        results = evaluator.evaluate(self.model, self.dataloader, verbose=False)

        # Check that results contain expected metrics
        self.assertIsInstance(results, dict)
        self.assertIn("mae", results)
        self.assertIn("mape", results)
        self.assertIn("rmse", results)
        self.assertIn("CRPS_GMM_GT", results)
        self.assertIn("mCCE", results)
        self.assertIn("mAW", results)

        # Check that all metrics are numeric
        for key, value in results.items():
            self.assertIsInstance(value, (int, float))
            self.assertFalse(np.isnan(value))

    def test_streaming_evaluator_matches_legacy_on_fixed_dataset(self):
        dataset = FixedDataset(num_samples=4, num_nodes=4, input_steps=3, pred_steps=2)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        model = DeterministicMetrGMMModel(
            pred_steps=dataset.pred_steps,
            num_mixtures=3,
        )

        with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as current_dir:
            legacy_evaluator = LegacyTwoPassMetrGMMEvaluator(
                save_dir=legacy_dir,
                gpu=False,
                ci_pts=31,
            )
            current_evaluator = MetrGMMEvaluator(
                save_dir=current_dir,
                gpu=False,
                ci_pts=31,
            )
            legacy_evaluator.eval_confs = [0.5, 0.9]
            current_evaluator.eval_confs = [0.5, 0.9]

            legacy_results = legacy_evaluator.evaluate(model, dataloader, verbose=False)
            current_results = current_evaluator.evaluate(model, dataloader, verbose=False)

            assert_metric_dict_allclose(self, current_results, legacy_results)
            assert_metric_dict_allclose(
                self, current_evaluator.metrics_scalar, legacy_evaluator.metrics_scalar
            )
            assert_metric_dict_allclose(
                self, current_evaluator.metrics_vector, legacy_evaluator.metrics_vector
            )

    def test_evaluate_calls_model_once_per_batch(self):
        dataloader = dummy_dataloader(
            batch_size=3,
            num_nodes=4,
            input_steps=2,
            pred_steps=2,
            num_samples=7,
        )
        model = CountingDummyModel(num_nodes=4, pred_steps=2, num_mixtures=2)
        evaluator = MetrGMMEvaluator(
            save_dir=self.temp_dir,
            gpu=False,
            ci_pts=11,
            data_max=10.0,
        )
        evaluator.eval_confs = [0.5]

        evaluator.evaluate(model, dataloader, verbose=False)

        self.assertEqual(model.forward_calls, len(dataloader))

    def test_evaluate_ignores_zero_and_nan_targets(self):
        dataset = InvalidTargetDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        evaluator = MetrGMMEvaluator(
            save_dir=self.temp_dir,
            gpu=False,
            ci_pts=21,
            data_max=10.0,
        )
        evaluator.eval_confs = [0.5]

        results = evaluator.evaluate(FixedPredictionGMMModel(), dataloader, verbose=False)

        expected_mape = ((2.0 / 3.0) + (2.0 / 5.0)) / 2.0
        np.testing.assert_allclose(results["mae"], 2.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(results["rmse"], 2.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(results["mape"], expected_mape, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(evaluator.metrics_vector["mae"], [2.0], rtol=1e-6, atol=1e-6)
        self.assertFalse(np.isnan(results["CRPS_GMM_GT"]))

    @unittest.skipUnless(os.path.exists("datasets/metr/test.npz"), "METR test data is not available")
    def test_matches_legacy_two_pass_evaluator_on_real_metr_subset(self):
        """Compare the production evaluator against the frozen two-pass baseline on real METR data."""
        dataset = MetrDataset(split="test")
        dataloader = DataLoader(
            Subset(dataset, range(4)),
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        model = DeterministicMetrGMMModel(
            pred_steps=dataset.pred_steps,
            num_mixtures=3,
        )

        with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as current_dir:
            legacy_evaluator = LegacyTwoPassMetrGMMEvaluator(
                save_dir=legacy_dir,
                gpu=False,
                ci_pts=51,
            )
            current_evaluator = MetrGMMEvaluator(
                save_dir=current_dir,
                gpu=False,
                ci_pts=51,
            )

            legacy_results = legacy_evaluator.evaluate(model, dataloader, verbose=False)
            current_results = current_evaluator.evaluate(model, dataloader, verbose=False)

            assert_metric_dict_allclose(self, current_results, legacy_results)
            assert_metric_dict_allclose(
                self, current_evaluator.metrics_scalar, legacy_evaluator.metrics_scalar
            )
            assert_metric_dict_allclose(
                self, current_evaluator.metrics_vector, legacy_evaluator.metrics_vector
            )

            legacy_evaluator.save_scores_to_json("scores.json")
            current_evaluator.save_scores_to_json("scores.json")
            with open(os.path.join(legacy_dir, "scores.json")) as f:
                legacy_saved = json.load(f)
            with open(os.path.join(current_dir, "scores.json")) as f:
                current_saved = json.load(f)

            self.assertEqual(set(current_saved.keys()), set(legacy_saved.keys()))
            assert_metric_dict_allclose(self, current_saved["average"], legacy_saved["average"])
            assert_metric_dict_allclose(self, current_saved["horizon"], legacy_saved["horizon"])


if __name__ == "__main__":
    unittest.main()
