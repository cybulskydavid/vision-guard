import torch
import torch.nn.functional as F
from torch import nn
import logging

logger = logging.getLogger(__name__)

class ProjectionLayerWrapper(nn.Module):
  def __init__(self, backbone: nn.Module, head: nn.Module):
    super().__init__()
    self.backbone = backbone
    self.backbone.requires_grad_(False)
    self.backbone.eval()
    self.head = head

  def forward(self, x: torch.Tensor):
    features = self.backbone(x)
    normalized_features = F.normalize(features)
    embeddings = self.head(normalized_features)
    return F.normalize(embeddings)

  def train(self, mode: bool = True):
    super().train(mode)
    if any(param.requires_grad for param in self.backbone.parameters()) and mode: self.backbone.train(); logger.debug("Backbone train")
    else: self.backbone.eval(); logger.debug("Backbone eval")
    return self