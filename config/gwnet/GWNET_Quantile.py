from skytraffic.config import LazyCall as L
from skytraffic.models import ForecastModel, GWNET_Quantile, QuantileTensorDataNormalizer
from skytraffic.data.datasets import MetrDataset

from ..common.train import train
from ..common.evaluation import metr_quantile_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .GWNET import dataset, dataloader

train.output_dir = "scratch/metr_gwnet_quantile"

quantile_levels = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
evaluator.quantile_levels = quantile_levels

model = L(ForecastModel)(
    network=L(GWNET_Quantile)(
        dropout=0.3,
        blocks=4,
        layers=2,
        gcn_bool=True,
        addaptadj=True,
        randomadj=True,
        aptonly=True,
        kernel_size=2,
        nhid=32,
        residual_channels=None,
        dilation_channels=None,
        skip_channels=None,
        end_channels=None,
        apt_layer=True,
        feature_dim=2,
        output_dim=1,
        quantiles=quantile_levels,
        input_steps=MetrDataset.input_steps,
        pred_steps=MetrDataset.pred_steps,
        num_nodes=MetrDataset.num_nodes,
        metadata="${..metadata}",
    ),
    normalizer=L(QuantileTensorDataNormalizer)(),
    data_null_value=MetrDataset.data_null_value,
    metadata=None,
)
