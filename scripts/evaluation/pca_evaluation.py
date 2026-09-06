import itertools

from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import MetricCollection
from torch import nn

from src.lightning.modules.pca_evaluation_module import PCAEvaluationModule
from src.modules.l2_norm_layer import L2NormLayer
from src.schemas.models import DinoResnet50, DinoVitS, DinoVitB, CLIPResnet50, CLIPVitB
from src.lightning.modules.evaluation_module import EvaluationModule
from src.data.class_subset import ClassSubset
from src.schemas.datasets import CUBDataset, CarsDataset, SOPDataset
from src.lightning.callbacks.leave_one_out_metrics_callback import LeaveOneOutMetricsCallback
from src.lightning.datamodules.class_data_module import ClassDataModule
from src.utils import cache_tools, path_creator
from src.utils.model_loader import load_model_and_transforms
from torchvision import datasets
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP
from torchvision.transforms import Compose

def main(model, dataset):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  batch_size=2048

  MODEL = model
  model, image_transforms, tensor_transforms = load_model_and_transforms(MODEL.name, device)

  DATASET = dataset
  full_dataset = datasets.ImageFolder(root=DATASET.path)
  transforms = Compose(image_transforms.transforms + tensor_transforms.transforms)
  train_dataset = ClassSubset(full_dataset, DATASET.train_classes, transforms)
  val_dataset = ClassSubset(full_dataset, DATASET.val_classes, transforms)
  test_dataset = ClassSubset(full_dataset, DATASET.test_classes, transforms)
  pca_dataset = ConcatDataset([train_dataset, val_dataset])

  pca_dataloader = DataLoader(pca_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

  projection_dim = 256
  pca_projection_save_path = f'pca/projection_layers/pca/{MODEL.name}/{DATASET.name}/{projection_dim}d.pt'
  pca_module = PCAEvaluationModule(model=model, save_path=pca_projection_save_path, target_dim=projection_dim)

  datamodule = ClassDataModule(
    batch_size=batch_size,
    num_workers=8,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset
  )

  test_metrics = MetricCollection({
    'Recall@1': RetrievalHitRate(top_k=1),
    'mAP': RetrievalMAP()
  })

  gallery_path = f'features/pca/{path_creator.create_path(MODEL.name, DATASET.name, output_dim=projection_dim, attack_name=None, margin=None, seed=None, iter_count=None)}'
  cache_path = f'cache/{gallery_path}'
  last_cached_file_idx = cache_tools.get_last_cached_batch_idx(cache_path)
  start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0
  trainer = Trainer(callbacks=[LeaveOneOutMetricsCallback(test_metrics, gallery_path, cache_path, start_idx)])
  trainer.predict(model=pca_module, dataloaders=pca_dataloader)

  projection_layer = nn.Linear(in_features=MODEL.out_features, out_features=projection_dim)
  saved_weights = torch.load(pca_projection_save_path)
  projection_layer.load_state_dict(saved_weights)
  projection_layer.requires_grad_(False)
  projection_layer.eval()
  model = nn.Sequential(model, L2NormLayer(dim=1), projection_layer,  L2NormLayer(dim=1))
  module = EvaluationModule(model=model)

  dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
  trainer.test(model=module, dataloaders=[dataloader])


if __name__ == '__main__':
  for model in [CLIPResnet50(), CLIPVitB(), DinoResnet50(), DinoVitB()]:
    for dataset in [CUBDataset(), CarsDataset(), SOPDataset()]:
      main(model, dataset)