from .GWNET import (
    train,
    model,
    optimizer,
    scheduler,
    dataset,
    dataloader,
)
from ..common.evaluation import metr_conformal_evaluator as evaluator

train.output_dir = "scratch/metr_gwnet_conformal"

evaluator.calibration_dataset = dataset.val
evaluator.calibration_batch_size = 32
