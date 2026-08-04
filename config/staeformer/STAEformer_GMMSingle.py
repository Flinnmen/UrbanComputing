from ..common.train import train
from ..common.evaluation import metr_gmm_evaluator as evaluator
from ..common.optim import AdamW as optimizer
from ..common.schedule import scheduler
from .STAEformer import dataset, dataloader
from .STAEformer_GMM import model

# Override train settings
train.output_dir = "scratch/metr_staeformer_gmm_single"
train.max_epoch = 20

model.network.anchors = [0.0]
model.network.sizes = [3.0]
