import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F

from src.attacks.base import BaseAttack
from src.attacks.cw import CWAttack

class AdversarialModule(pl.LightningModule):
  def __init__(self, model: nn.Module, attack: BaseAttack, tensor_transforms, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.attack = attack
    self.tensor_transforms = tensor_transforms


  def forward(self, images):
    images = self.tensor_transforms(images)
    features = self.model(images)
    return F.normalize(features)
  

  def test_step(self, batch, batch_idx, dataloader_idx=0):
    images, labels, images_idx, images_name = batch

    adv_images = self.attack(self, images, images_idx, self.device)
    with torch.no_grad():      
      adv_features = self(adv_images)

    adv_images = adv_images if type(self.attack).__name__ == CWAttack.__name__ else None

    return {'features': adv_features, 'labels': labels, 'images_idx': images_idx, 'images_name': images_name, 'adv_images': adv_images}