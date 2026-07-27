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
train.output_dir = "scratch/metr_gwnet_gmm_anchor7"

model.network.anchors = [-2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25]
model.network.sizes = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
