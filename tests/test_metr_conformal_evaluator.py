import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from skytraffic.evaluation.metr_uncertainty_evaluation import MetrConformalEvaluator


class FixedTargetDataset(Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return {
            "source": torch.zeros(1),
            "target": self.targets[idx],
        }

    def collate_fn(self, batch):
        return {
            "source": torch.stack([item["source"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
        }


class ZeroPredictionModel(nn.Module):
    def forward(self, data_dict):
        return {"pred": torch.zeros_like(data_dict["target"])}


class TestMetrConformalEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _expected_abs_quantiles(self, targets, conf):
        expected = torch.empty(targets.shape[1], targets.shape[2])
        for horizon in range(targets.shape[1]):
            for place in range(targets.shape[2]):
                residuals = targets[:, horizon, place].abs()
                q_level = min(1.0, np.ceil((residuals.numel() + 1) * conf) / residuals.numel())
                expected[horizon, place] = torch.quantile(residuals, q_level)
        return expected

    def test_calibration_residuals_are_separate_per_place(self):
        targets = torch.tensor([
            [[1.0, 10.0], [2.0, 20.0]],
            [[2.0, 20.0], [4.0, 40.0]],
            [[3.0, 30.0], [6.0, 60.0]],
            [[4.0, 40.0], [8.0, 80.0]],
        ])
        dataset = FixedTargetDataset(targets)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        evaluator = MetrConformalEvaluator(
            save_dir=self.temp_dir,
            calibration_dataloader=dataloader,
            max_residual_samples=4,
        )
        evaluator.eval_confs = [0.5]

        _, _, abs_quantiles = evaluator._calibrate(ZeroPredictionModel())
        expected_half_width = self._expected_abs_quantiles(targets, conf=0.5)

        np.testing.assert_allclose(abs_quantiles[0.5].numpy(), expected_half_width.numpy())
        self.assertNotEqual(abs_quantiles[0.5][0, 0].item(), abs_quantiles[0.5][0, 1].item())

        results = evaluator.evaluate(ZeroPredictionModel(), dataloader, verbose=False)

        expected_width_by_horizon = (2.0 * expected_half_width).mean(dim=1)
        np.testing.assert_allclose(
            evaluator.metrics_vector["CI_WIDTH_0.5"],
            expected_width_by_horizon.numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            results["CI_WIDTH_0.5"],
            expected_width_by_horizon.mean().item(),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
