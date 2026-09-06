import itertools

from pytorch_lightning import Trainer
import torch
from torchmetrics import MetricCollection

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

  MODEL = model
  model, image_transforms, tensor_transforms = load_model_and_transforms(MODEL.name, device)
  model = EvaluationModule(model)

  DATASET = dataset
  full_dataset = datasets.ImageFolder(root=DATASET.path)
  transforms = Compose(image_transforms.transforms + tensor_transforms.transforms)
  train_dataset = ClassSubset(full_dataset, DATASET.train_classes, transforms)
  val_dataset = ClassSubset(full_dataset, DATASET.val_classes, transforms)
  test_dataset = ClassSubset(full_dataset, DATASET.test_classes, transforms)

  datamodule = ClassDataModule(
    batch_size=2048,
    num_workers=8,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset
  )

  test_metrics = MetricCollection({
    'Recall@1': RetrievalHitRate(top_k=1),
    'mAP': RetrievalMAP()
  })

  out_dim = MODEL.out_features
  gallery_path = path_creator.create_path(MODEL.name, DATASET.name, output_dim=out_dim, attack_name=None, margin=None, seed=None, iter_count=None)
  cache_path = f'cache/{gallery_path}'
  last_cached_file_idx = cache_tools.get_last_cached_batch_idx(cache_path)
  start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0
  trainer = Trainer(callbacks=[LeaveOneOutMetricsCallback(test_metrics, gallery_path, cache_path, start_idx)])
  
  dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
  trainer.test(model=model, dataloaders=[dataloader])


if __name__ == '__main__':
  for model in [DinoResnet50(), DinoVitB(), CLIPResnet50(), CLIPVitB()]:
    for dataset in [CUBDataset(), CarsDataset(), SOPDataset()]:
      main(model, dataset)