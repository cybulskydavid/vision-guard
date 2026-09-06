import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F

class EvaluationModule(pl.LightningModule):
  def __init__(self, model: nn.Module, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model

  
  def forward(self, images):
    features = self.model(images)
    return F.normalize(features)
  

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    images, labels, images_idx, images_name = batch

    with torch.no_grad():
      features = self(images)

      return {'features': features, 'labels': labels, 'images_idx': images_idx, 'images_name': images_name}