import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners


class ProjectionLayerModule(pl.LightningModule):
  def __init__(
    self,
    model,
    loss_fn,
    miner=None,
    lr=1e-4,
    weight_decay=1e-2,
    max_epochs=100,
    optimizer_cls=torch.optim.AdamW,
  ):
    super().__init__()
    self.save_hyperparameters(ignore=["model", "loss_fn", "miner"])
    self.model = model
    self.loss_fn = loss_fn
    self.miner = miner
    self.lr = lr
    self.weight_decay = weight_decay
    self.max_epochs = max_epochs
    self.optimizer_cls = optimizer_cls

  def forward(self, x):
    return self.model(x)

  def on_train_epoch_start(self):
    self.model.train()

  def training_step(self, batch, batch_idx, dataloader_idx=0):
    x, labels, _, _ = batch
    embeddings = self(x)

    if self.miner is not None:
      mined = self.miner(embeddings, labels)
      loss = self.loss_fn(embeddings, labels, mined)
    else:
      loss = self.loss_fn(embeddings, labels)

    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    x, labels, _, _ = batch
    with torch.no_grad():
      embeddings = self(x)
    return {"embeddings": embeddings.detach(), "labels": labels.detach()}

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    x, labels, _, _ = batch
    with torch.no_grad():
      embeddings = self(x)
    return {"embeddings": embeddings.detach(), "labels": labels.detach()}

  def configure_optimizers(self):
    head_params = self.model.head.parameters()
    loss_params = self.loss_fn.parameters() if isinstance(self.loss_fn, nn.Module) else []

    optimizer = self.optimizer_cls(
      [{"params": list(head_params) + list(loss_params), "lr": self.lr}],
      weight_decay=self.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer=optimizer, T_max=self.max_epochs
    )
    return {
      "optimizer": optimizer,
      "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1},
    }
