import itertools
from pathlib import Path
import statistics

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import LpDistance
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP
from torchvision import datasets
from torchvision.transforms import Compose

from src.data.class_subset import ClassSubset
from src.lightning.callbacks.leave_one_out_metrics_callback import LeaveOneOutMetricsCallback
from src.lightning.datamodules.class_data_module import ClassDataModule
from src.lightning.modules.evaluation_module import EvaluationModule
from src.modules.l2_norm_layer import L2NormLayer
from src.schemas.datasets import CUBDataset, CarsDataset, SOPDataset
from src.schemas.models import CLIPResnet50, CLIPVitB, DinoResnet50, DinoVitB
from src.utils import cache_tools, path_creator
from src.utils.model_loader import load_model_and_transforms

from src.modules.projection_layer_wrapper import ProjectionLayerWrapper
from src.lightning.modules.projection_layer_module import ProjectionLayerModule
from src.lightning.callbacks.leave_one_out_validation_metrics_callback import LeaveOneOutValidationMetricsCallback


LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 100
P, K = 16, 8
ARCFACE_SCALE = 32.0

TRIPLET_DISTANCE = LpDistance(normalize_embeddings=False, p=2, power=1)
TRIPLET_MARGIN_GRID = [0.1]
ARCFACE_MARGIN_GRID = [0.5]
FINAL_SEEDS = [31, 90, 29, 13, 37]
TUNING_SEEDS = [31, 90, 29]

PROJECTION_DIM = 256

def get_targets(dataset) -> list:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = dataset.dataset.targets
        return [base_targets[i] for i in dataset.indices]
    return [label for _, label, _, _ in dataset]



def make_train_dataloader(train_dataset):
    targets = get_targets(train_dataset)
    sampler = samplers.MPerClassSampler(
        labels=targets, m=K, batch_size=P * K, length_before_new_iter=len(train_dataset)
    )
    return DataLoader(train_dataset, batch_size=P * K, sampler=sampler, drop_last=True, num_workers=8)
 
 
def build_loss(loss_type: str, margin: float, num_classes: int = None):
    if loss_type == "triplet":
        loss_fn = losses.TripletMarginLoss(margin=margin, distance=TRIPLET_DISTANCE)
        miner = miners.TripletMarginMiner(margin=margin, distance=TRIPLET_DISTANCE, type_of_triplets="semihard")
        return loss_fn, miner
    elif loss_type == "arcface":
        assert num_classes is not None
        loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes, embedding_size=PROJECTION_DIM, margin=margin, scale=ARCFACE_SCALE
        )
        return loss_fn, None
    raise ValueError(loss_type)
 
 
def train_one_configuration(model_schema, dataset_schema, loss_type: str, margin: float, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(seed, workers=True)
 
    backbone, image_transforms, tensor_transforms = load_model_and_transforms(model_schema.name, device)
    full_dataset = datasets.ImageFolder(root=dataset_schema.path)
    transforms = Compose(image_transforms.transforms + tensor_transforms.transforms)
 
    train_dataset = ClassSubset(full_dataset, dataset_schema.train_classes, transforms)
    val_dataset = ClassSubset(full_dataset, dataset_schema.val_classes, transforms)
 
    num_classes = len(dataset_schema.train_classes)
 
    head = nn.Linear(model_schema.out_features, PROJECTION_DIM)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
 
    wrapper = ProjectionLayerWrapper(backbone=backbone, head=head)
    loss_fn, miner = build_loss(loss_type, margin, num_classes if loss_type == "arcface" else None)
 
    module = ProjectionLayerModule(
        model=wrapper,
        loss_fn=loss_fn,
        miner=miner,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS
    )
 
    train_loader = make_train_dataloader(train_dataset)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=8)
 
    csv_logger = CSVLogger(
        save_dir="logs",
        name=f"{loss_type}/{model_schema.name}/{dataset_schema.name}",
        version=f"margin{margin}_seed{seed}",
    )
 
    val_metrics_cb = LeaveOneOutValidationMetricsCallback()
    checkpoint_cb = ModelCheckpoint(monitor="val_mAP", mode="max", save_top_k=1, filename="best-{epoch}-{val_mAP:.4f}")
    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")
    # early_stop_cb = EarlyStopping(monitor="val_mAP", patience=10, mode="max", min_delta=0.00, verbose=True)       
 
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        logger=csv_logger,
        callbacks=[val_metrics_cb, checkpoint_cb, lr_monitor_cb]
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
 
    best_val_map = checkpoint_cb.best_model_score.item()
 
    projection_save_path = (
        f"metric_learning/projection_layers/{loss_type}/{model_schema.name}/"
        f"{dataset_schema.name}/{PROJECTION_DIM}d_margin{margin}_seed{seed}.pt"
    )
    Path(f"metric_learning/projection_layers/{loss_type}/{model_schema.name}/"
            f"{dataset_schema.name}").mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.head.state_dict(), projection_save_path)
 
    return {
        "loss_type": loss_type,
        "margin": margin,
        "seed": seed,
        "val_mAP": best_val_map,
        "projection_path": projection_save_path,
    }
 
 
def evaluate_on_test(model_schema, dataset_schema, trained_result: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2048
 
    backbone, image_transforms, tensor_transforms = load_model_and_transforms(model_schema.name, device)
    full_dataset = datasets.ImageFolder(root=dataset_schema.path)
    transforms = Compose(image_transforms.transforms + tensor_transforms.transforms)
 
    train_dataset = ClassSubset(full_dataset, dataset_schema.train_classes, transforms)
    val_dataset = ClassSubset(full_dataset, dataset_schema.val_classes, transforms)
    test_dataset = ClassSubset(full_dataset, dataset_schema.test_classes, transforms)
 
    datamodule = ClassDataModule(
        batch_size=batch_size, num_workers=8,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
    )
 
    projection_layer = nn.Linear(in_features=model_schema.out_features, out_features=PROJECTION_DIM)
    projection_layer.load_state_dict(torch.load(trained_result["projection_path"]))
    projection_layer.requires_grad_(False)
    projection_layer.eval()
 
    eval_model = nn.Sequential(backbone, L2NormLayer(dim=1), projection_layer, L2NormLayer(dim=1))
    module = EvaluationModule(model=eval_model)
 
    test_metrics = MetricCollection({"Recall@1": RetrievalHitRate(top_k=1), "mAP": RetrievalMAP()})
 
    gallery_path = (
        f"features/{trained_result['loss_type']}/"
        f"{path_creator.create_path(model_schema.name, dataset_schema.name, output_dim=PROJECTION_DIM, attack_name=None, margin=trained_result['margin'], seed=trained_result['seed'], iter_count=None)}"
    )
    cache_path = f"cache/{gallery_path}"
    last_cached_file_idx = cache_tools.get_last_cached_batch_idx(cache_path)
    start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0
 
    trainer = Trainer(callbacks=[LeaveOneOutMetricsCallback(test_metrics, gallery_path, cache_path, start_idx)])
    dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
    return trainer.test(model=module, dataloaders=[dataloader])
 
 
def select_best_margin(model_schema, dataset_schema, loss_type: str, margin_grid: list):
    margin_to_results = {}
    for margin in margin_grid:
        seed_results = [
            train_one_configuration(model_schema, dataset_schema, loss_type, margin, seed)
            for seed in TUNING_SEEDS
        ]
        mean_val_map = statistics.mean(r["val_mAP"] for r in seed_results)
        margin_to_results[margin] = {
            "mean_val_mAP": mean_val_map,
            "seed_results": seed_results,
        }

    best_margin = max(margin_to_results, key=lambda m: margin_to_results[m]["mean_val_mAP"])
    best_mean = margin_to_results[best_margin]["mean_val_mAP"]
    print(
        f"[{loss_type}] {model_schema.name}/{dataset_schema.name} -> margines={best_margin} "
        f"(srednie val_mAP z {len(TUNING_SEEDS)} seedow={best_mean:.4f})"
    )
    return best_margin, margin_to_results

def _aggregate_test_metrics(per_seed_results: list) -> dict:
    metric_values: dict = {}
    for r in per_seed_results:
        test_res = r["test_metrics"]
        result_dict = test_res[0] if isinstance(test_res, list) else test_res
        for key, value in result_dict.items():
            metric_values.setdefault(key, []).append(float(value))

    aggregated = {}
    for key, values in metric_values.items():
        aggregated[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregated 
 
def run_final_and_evaluate(model_schema, dataset_schema, loss_type: str, best_margin: float):
    per_seed_results = []
    for seed in FINAL_SEEDS:
        trained = train_one_configuration(model_schema, dataset_schema, loss_type, best_margin, seed)
        test_res = evaluate_on_test(model_schema, dataset_schema, trained)
        per_seed_results.append({
            "seed": seed,
            "val_mAP": trained["val_mAP"],
            "projection_path": trained["projection_path"],
            "test_metrics": test_res,
        })

    val_maps = [r["val_mAP"] for r in per_seed_results]
    mean_val_mAP = statistics.mean(val_maps)
    std_val_mAP = statistics.stdev(val_maps) if len(val_maps) > 1 else 0.0

    aggregated_test = _aggregate_test_metrics(per_seed_results)

    median_val_mAP = statistics.median(val_maps)
    median_seed_result = min(per_seed_results, key=lambda r: abs(r["val_mAP"] - median_val_mAP))

    print(
        f"[{loss_type}] {model_schema.name}/{dataset_schema.name} margines={best_margin} -> "
        f"val_mAP = {mean_val_mAP:.4f} +/- {std_val_mAP:.4f} (n={len(FINAL_SEEDS)}); "
        f"model medianowy: seed={median_seed_result['seed']}"
    )

    return {
        "per_seed_results": per_seed_results,
        "val_mAP_mean": mean_val_mAP,
        "val_mAP_std": std_val_mAP,
        "test_metrics_aggregated": aggregated_test,
        "median_seed": median_seed_result["seed"],
        "median_seed_projection_path": median_seed_result["projection_path"],
    }

 
if __name__ == "__main__":
    all_results = {}

    for model in [CLIPVitB()]:
        for dataset in [CarsDataset()]:
            for loss_type, grid in [("triplet", TRIPLET_MARGIN_GRID), ("arcface", ARCFACE_MARGIN_GRID)]:
                print(f"\n===== {model.name} / {dataset.name} / {loss_type} =====")
                best_margin, tuning_results = select_best_margin(model, dataset, loss_type, grid)
                final_results = run_final_and_evaluate(model, dataset, loss_type, best_margin)
                all_results[(model.name, dataset.name, loss_type)] = {
                    "best_margin": best_margin,
                    "tuning_results": tuning_results,
                    "final_results": final_results,
                }

    torch.save(all_results, "metric_learning_all_results.pt")
 