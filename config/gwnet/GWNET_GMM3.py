from .GWNET_GMM import (
    train,
    model,
    optimizer,
    scheduler,
    dataloader,
    dataset,
    evaluator,
)

# Override train settings
train.output_dir = "scratch/metr_gwnet_gmm_anchor3"

model.network.anchors = [-1.5, 0.0, 1.5]
model.network.sizes = [1.5, 1.5, 1.5]
