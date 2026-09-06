import os

import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F

class PCAEvaluationModule(pl.LightningModule):
  def __init__(self, model: nn.Module, save_path: str, target_dim: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.save_path = save_path
    self.target_dim = target_dim

    self.model.eval()
    for param in self.model.parameters():
      param.requires_grad = False

    self.extracted_features = []
    self.pca_projection = None

  def forward(self, x):
    features = F.normalize(self.model(x))
    if self.pca_projection is not None:
      return F.normalize(self.pca_projection(features))
    return features
  
  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    x, _, _, _ = batch

    with torch.no_grad():
      features = self(x)
      self.extracted_features.append(features.cpu())

    return features
  

  def on_predict_epoch_end(self):
    all_features = torch.cat(self.extracted_features)
    all_features = F.normalize(all_features, p=2, dim=1)
    in_dim = all_features.shape[1]

    print(f"Total dataset shape: {all_features.shape}")
    feature_mean = torch.mean(all_features, dim=0, keepdim=True)
    centered_features = all_features - feature_mean

    print(f"Calculating PCA to reduce {in_dim}D -> {self.target_dim}D...")
    _, _, V = torch.pca_lowrank(centered_features, q=self.target_dim, center=False)

    pca_bias = -torch.matmul(feature_mean, V).squeeze()

    self.pca_projection = nn.Linear(in_dim, self.target_dim)
    with torch.no_grad():
      self.pca_projection.weight.copy_(V.T)
      self.pca_projection.bias.copy_(pca_bias)

      self.pca_projection.requires_grad_(False)
      self.pca_projection.to(self.device)

    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    torch.save(self.pca_projection.state_dict(), self.save_path)
    print(f"Projection layer successfully saved to: {self.save_path}")
    self.extracted_features.clear()