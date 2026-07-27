from .GWNET import (
    train,
    model,
    optimizer,
    scheduler,
    dataset,
    dataloader,
)
from ..common.evaluation import metr_mc_dropout_evaluator as evaluator

train.output_dir = "scratch/metr_gwnet_mc_dropout"
