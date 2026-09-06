import hydra
from omegaconf import DictConfig, OmegaConf

import itertools
import random
import re
import statistics
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer
import torch
from torch import nn
from torchmetrics import MetricCollection

from src.attacks.cw import CWAttack
from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack

from src.lightning.modules.adversarial_module import AdversarialModule
from src.modules.l2_norm_layer import L2NormLayer
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

FINAL_SEEDS = [90, 37]
PROJECTION_LAYERS_ROOT = Path("metric_learning/projection_layers")
LOGS_ROOT = Path("logs")

PROJECTION_FILE_RE = re.compile(r"^(?P<dim>\d+)d_margin(?P<margin>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
CHECKPOINT_VAL_MAP_RE = re.compile(r"val_mAP=(?P<val_map>\d+\.\d+)")


def discover_margin_seed_files(loss_type: str, model_name: str, dataset_name: str) -> dict:
    directory = PROJECTION_LAYERS_ROOT / loss_type / model_name / dataset_name
    if not directory.is_dir():
        return {}

    margin_to_seed_paths: dict = {}
    for file in directory.glob("*.pt"):
        match = PROJECTION_FILE_RE.match(file.name)
        if not match:
            continue
        margin = float(match.group("margin"))
        seed = int(match.group("seed"))
        margin_to_seed_paths.setdefault(margin, {})[seed] = file
    return margin_to_seed_paths


def read_val_map_from_checkpoint(loss_type: str, model_name: str, dataset_name: str, margin: float, seed: int):
    checkpoints_dir = LOGS_ROOT / loss_type / model_name / dataset_name / f"margin{margin}_seed{seed}" / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    for ckpt_file in checkpoints_dir.glob("best-*.ckpt"):
        match = CHECKPOINT_VAL_MAP_RE.search(ckpt_file.name)
        if match:
            return float(match.group("val_map"))
    return None


def select_best_margin(loss_type: str, model_name: str, dataset_name: str, margin_to_seed_paths: dict) -> float:
    candidates = [
        margin for margin, seed_paths in margin_to_seed_paths.items()
        if set(FINAL_SEEDS).issubset(seed_paths.keys())
    ]
    if not candidates:
        available = {m: sorted(s.keys()) for m, s in margin_to_seed_paths.items()}
        raise ValueError(
            f"Żaden margines dla {loss_type}/{model_name}/{dataset_name} nie ma zapisanych "
            f"wszystkich finałowych seedów {FINAL_SEEDS}. Dostępne marginesy/seedy: {available}"
        )
    if len(candidates) == 1:
        return candidates[0]

    def mean_val_map(margin):
        values = [
            v for v in (
                read_val_map_from_checkpoint(loss_type, model_name, dataset_name, margin, seed)
                for seed in FINAL_SEEDS
            ) if v is not None
        ]
        return statistics.mean(values) if values else float("-inf")

    return max(candidates, key=mean_val_map)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_attack(attack_type: str, margin: float, iter_count: int, attack_seed: int, clean_data: dict):
    common_kwargs = dict(
        mining_function=mining.positives_negatives_miner,
        gallery=clean_data['embeddings'],
        labels=clean_data['labels'],
        n_positives=1,
        n_negatives=1,
    )
    if attack_type == "fgsm":
        return FGSMAttack(epsilon=margin, seed=attack_seed, **common_kwargs)
    elif attack_type == "pgd":
        return PGDAttack(
            epsilon=margin, alpha=(2.5 * margin) / iter_count, iter_count=iter_count,
            num_restarts=1, seed=attack_seed, **common_kwargs
        )
    elif attack_type == "cw":
        return CWAttack(kappa=margin, iter_count=iter_count, learning_rate=1e-2, **common_kwargs)
    else:
        raise ValueError(f"Nieznany attack_type: {attack_type!r} (oczekiwano 'fgsm', 'pgd' lub 'cw')")


def main(model, dataset, loss_type, train_margin, margin, model_seed, attack_seed, attack_type="cw", projection_dim=256):
    logger.info(f"loss_type: {loss_type}, train_margin (margines treningowy modelu): {train_margin}")
    logger.info(f"margin (budzet ataku): {margin}")
    logger.info(f"model_seed (identyfikuje wytrenowana warstwe projekcji/cechy): {model_seed}")
    logger.info(f"attack_seed (losowosc samego ataku, niezalezna od treningu): {attack_seed}")
    seed_everything(attack_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MODEL = model
    DATASET = dataset
    logger.info(f'model {MODEL.name}')
    model, image_transforms, tensor_transforms = load_model_and_transforms(MODEL.name, device)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval() 

    metric_projection_save_path = (
        f'metric_learning/projection_layers/{loss_type}/{MODEL.name}/{DATASET.name}/'
        f'{projection_dim}d_margin{train_margin}_seed{model_seed}.pt'
    )
    logger.info(f"Loading metric learning projection layer from: {metric_projection_save_path}")

    projection_layer = nn.Linear(in_features=MODEL.out_features, out_features=projection_dim)
    projection_layer.load_state_dict(torch.load(metric_projection_save_path, map_location=device))
    projection_layer.requires_grad_(False)
    projection_layer.eval()

    full_model = nn.Sequential(model, L2NormLayer(dim=1), projection_layer,  L2NormLayer(dim=1))
    full_model.eval()

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

    out_dim = projection_dim
    iter_count = 100

    base_gallery_path = path_creator.create_path(
        MODEL.name, DATASET.name, output_dim=out_dim, attack_name=None, iter_count=None, margin=train_margin, seed=model_seed
    )
    
    gallery_path = f'features/{loss_type}/{base_gallery_path}/features.pt'
    clean_data = torch.load(gallery_path)

    attack = build_attack(attack_type, margin, iter_count, attack_seed, clean_data)

    base_adv_path = path_creator.create_path(
        MODEL.name, DATASET.name, output_dim=out_dim, attack_name=type(attack).__name__, iter_count=iter_count, margin=attack.margin, seed=model_seed
    )
    
    adv_features_save_path = (
        f'features/{loss_type}/{base_adv_path}/{attack_seed}'
    )
    
    adv_features_cache_path = f'cache/{adv_features_save_path}'
    last_cached_file_idx = cache_tools.get_last_cached_batch_idx(adv_features_cache_path)
    start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0
    
    trainer = Trainer(callbacks=[AdvLeaveOneOutMetricsCallback(save_path=adv_features_save_path, cache_path=adv_features_cache_path, gallery_embeddings=clean_data['embeddings'], gallery_labels=clean_data['labels'], gallery_idx=clean_data['images_idx'], test_metrics=test_metrics, batch_offset=start_idx)])
    
    module = AdversarialModule(model=full_model, attack=attack, tensor_transforms=tensor_transforms)
    
    dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
    trainer.test(model=module, dataloaders=[dataloader])

if __name__ == '__main__':
    for model in [DinoVitB()]:
        for dataset in [CarsDataset()]:
            for loss_type in ['arcface']:
                margin_to_seed_paths = discover_margin_seed_files(loss_type, model.name, dataset.name)
                if not margin_to_seed_paths:
                    logger.info(
                        f"Brak zapisanych warstw projekcji dla {loss_type}/{model.name}/{dataset.name} - pomijam."
                    )
                    continue

                best_train_margin = select_best_margin(loss_type, model.name, dataset.name, margin_to_seed_paths)
                model_seeds = sorted(margin_to_seed_paths[best_train_margin].keys())
                logger.info(
                    f"{loss_type}/{model.name}/{dataset.name} -> najlepszy margines treningowy: "
                    f"{best_train_margin} (seedy modeli: {model_seeds})"
                )
                attack_seeds = [31]

                for attack_type in ['cw']:
                    for margin in [0.5]:
                        for model_seed in model_seeds:
                            for attack_seed in attack_seeds:
                                main(
                                    model, dataset, loss_type, best_train_margin, margin,
                                    model_seed, attack_seed, attack_type=attack_type, projection_dim=256
                                )