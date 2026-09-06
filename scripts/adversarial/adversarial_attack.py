import hydra
from omegaconf import DictConfig, OmegaConf

import itertools
import random

import numpy as np
from pytorch_lightning import Trainer
import torch
from torchmetrics import MetricCollection

from src.attacks.cw import CWAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack

from src.lightning.modules.adversarial_module import AdversarialModule
from src.schemas.models import DinoResnet50, DinoVitB, CLIPResnet50, CLIPVitB
from src.lightning.modules.evaluation_module import EvaluationModule
from src.data.class_subset import ClassSubset
from src.schemas.datasets import CUBDataset, CarsDataset, SOPDataset
from src.lightning.callbacks.adv_leave_one_out_metrics_callback import AdvLeaveOneOutMetricsCallback
from src.lightning.datamodules.class_data_module import ClassDataModule
from src.utils import cache_tools, mining, path_creator
from src.utils.logger import setup_logger
from src.utils.model_loader import load_model_and_transforms
from torchvision import datasets
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP
from torchvision.transforms import Compose

logger = setup_logger(__name__)

def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)

  torch.manual_seed(seed)

  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def main(model, dataset, margin, seed):
  logger.info(f"margin: {margin}")
  SEED = seed
  logger.info(f'seed {SEED}')
  seed_everything(SEED)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  MODEL = model
  logger.info(f'model {MODEL.name}')
  model, image_transforms, tensor_transforms = load_model_and_transforms(MODEL.name, device)

  for param in model.parameters():
    param.requires_grad_(False)

  DATASET = dataset
  logger.info(f'dataset {DATASET.name}')
  full_dataset = datasets.ImageFolder(root=DATASET.path)
  train_dataset = ClassSubset(full_dataset, DATASET.train_classes, image_transforms)
  val_dataset = ClassSubset(full_dataset, DATASET.val_classes, image_transforms)
  test_dataset = ClassSubset(full_dataset, DATASET.test_classes, image_transforms)

  datamodule = ClassDataModule(
    batch_size=256,
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
  # iter_count = None
  iter_count = 100
  gallery_path = f'{path_creator.create_path(MODEL.name, DATASET.name, output_dim=out_dim, attack_name=None, iter_count=iter_count, margin=None, seed=SEED)}/features.pt'
  clean_data = torch.load(gallery_path)

  # attack = PGDAttack(epsilon=margin, alpha=(2.5*margin)/iter_count, iter_count=iter_count, num_restarts=1, mining_function=mining.positives_negatives_miner, gallery=clean_data['embeddings'], labels=clean_data['labels'], n_positives=1, n_negatives=1)
  # attack = FGSMAttack(epsilon=margin, mining_function=mining.positives_negatives_miner, gallery=clean_data['embeddings'], labels=clean_data['labels'], n_positives=1, n_negatives=1)
  attack = CWAttack(kappa=margin, iter_count=iter_count, learning_rate=1e-2, mining_function=mining.positives_negatives_miner, gallery=clean_data['embeddings'], labels=clean_data['labels'], n_positives=1, n_negatives=1)

  adv_features_save_path = path_creator.create_path(MODEL.name, DATASET.name, output_dim=out_dim, attack_name=type(attack).__name__, iter_count=iter_count, margin=attack.margin, seed=SEED)
  adv_features_cache_path = f'cache/{adv_features_save_path}'
  last_cached_file_idx = cache_tools.get_last_cached_batch_idx(adv_features_cache_path)
  start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0
  trainer = Trainer(callbacks=[AdvLeaveOneOutMetricsCallback(save_path=adv_features_save_path, cache_path=adv_features_cache_path, gallery_embeddings=clean_data['embeddings'], gallery_labels=clean_data['labels'], gallery_idx=clean_data['images_idx'], test_metrics=test_metrics, batch_offset=start_idx)])
  module = AdversarialModule(model=model,attack=attack,tensor_transforms=tensor_transforms)
  
  dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, start_idx+8)
  model.requires_grad_(False)
  trainer.test(model=module, dataloaders=[dataloader])


if __name__ == '__main__':
  for model in [CLIPVitB(), CLIPResnet50(), DinoVitB(), DinoResnet50()]:
    for dataset in [CUBDataset(), CarsDataset(), SOPDataset()]:
      for margin in [1/255, 2/255, 4/255, 8/255]:
      # for margin in [0]:
        for seed in [31, 90, 29]:
          main(model, dataset, margin, seed)
