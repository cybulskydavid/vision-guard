import itertools
from pathlib import Path
import statistics

import torch
from pytorch_lightning import Trainer
from torch import nn
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


PROJECTION_DIM = 256

# Plik wygenerowany przez wcześniejszy (treningowy) przebieg skryptu.
# Zawiera dla każdej konfiguracji (model, dataset, loss_type) wybrany
# best_margin oraz per_seed_results ze ścieżkami do zapisanych warstw
# projekcyjnych (projection_path).
RESULTS_INPUT_PATH = "metric_learning_all_results.pt"
RESULTS_OUTPUT_PATH = "metric_learning_all_results_eval.pt"


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

    base_gallery_path = path_creator.create_path(
        model_schema.name,
        dataset_schema.name,
        output_dim=PROJECTION_DIM,
        attack_name=None,
        margin=trained_result["margin"],
        seed=trained_result["seed"],
        iter_count=None,
    )
    gallery_path = (
        f"features/{trained_result['loss_type']}/{base_gallery_path}"
    )
    cache_path = f"cache/{gallery_path}"
    last_cached_file_idx = cache_tools.get_last_cached_batch_idx(cache_path)
    start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0

    trainer = Trainer(callbacks=[LeaveOneOutMetricsCallback(test_metrics, gallery_path, cache_path, start_idx)])
    dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
    return trainer.test(model=module, dataloaders=[dataloader])


def evaluate_saved_configuration(model_schema, dataset_schema, loss_type: str, best_margin: float, per_seed_results: list) -> list:
    """Uruchamia ewaluację na zbiorze testowym dla już wytrenowanych modeli
    (bez ponownego treningu) - wykorzystuje zapisane projection_path."""
    updated_seed_results = []
    for seed_result in per_seed_results:
        trained_result = {
            "loss_type": loss_type,
            "margin": best_margin,
            "seed": seed_result["seed"],
            "projection_path": seed_result["projection_path"],
        }
        test_metrics = evaluate_on_test(model_schema, dataset_schema, trained_result)
        updated_seed_results.append({**seed_result, "test_metrics": test_metrics})
    return updated_seed_results


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


if __name__ == "__main__":
    all_results = torch.load(RESULTS_INPUT_PATH)
    model_schemas = {m.name: m for m in [CLIPResnet50(), CLIPVitB(), DinoResnet50(), DinoVitB()]}
    dataset_schemas = {d.name: d for d in [CUBDataset(), CarsDataset()]}

    for (model_name, dataset_name, loss_type), entry in all_results.items():
        model_schema = model_schemas[model_name]
        dataset_schema = dataset_schemas[dataset_name]
        best_margin = entry["best_margin"]
        per_seed_results = entry["final_results"]["per_seed_results"]

        print(f"\n===== EWALUACJA: {model_name} / {dataset_name} / {loss_type} (margines={best_margin}) =====")

        # updated_seed_results = evaluate_saved_configuration(
        #     model_schema, dataset_schema, loss_type, best_margin, per_seed_results
        # )
        # aggregated_test = _aggregate_test_metrics(updated_seed_results)

        # val_maps = [r["val_mAP"] for r in updated_seed_results]
        # median_val_mAP = statistics.median(val_maps)
        # median_seed_result = min(updated_seed_results, key=lambda r: abs(r["val_mAP"] - median_val_mAP))

        # entry["final_results"]["per_seed_results"] = updated_seed_results
        # entry["final_results"]["test_metrics_aggregated"] = aggregated_test
        # entry["final_results"]["median_seed"] = median_seed_result["seed"]
        # entry["final_results"]["median_seed_projection_path"] = median_seed_result["projection_path"]

        # for metric_name, stats in aggregated_test.items():
        #     print(f"[{loss_type}] {model_name}/{dataset_name} -> {metric_name} = {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # torch.save(all_results, RESULTS_OUTPUT_PATH)