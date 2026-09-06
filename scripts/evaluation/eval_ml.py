import argparse
import itertools
import re
import statistics
from pathlib import Path

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
from src.modules.projection_layer_wrapper import ProjectionLayerWrapper
from src.schemas.datasets import CUBDataset, CarsDataset, SOPDataset
from src.schemas.models import CLIPResnet50, CLIPVitB, DinoResnet50, DinoVitB
from src.utils import cache_tools, path_creator
from src.utils.model_loader import load_model_and_transforms


PROJECTION_DIM = 256
FINAL_SEEDS = [31, 90, 29, 13, 37]  # musi się zgadzać ze skryptem treningowym

PROJECTION_LAYERS_ROOT = Path("metric_learning/projection_layers")
LOGS_ROOT = Path("logs")

MODEL_SCHEMAS = {m.name: m for m in [CLIPResnet50(), CLIPVitB(), DinoResnet50(), DinoVitB()]}
DATASET_SCHEMAS = {d.name: d for d in [CUBDataset(), CarsDataset(), SOPDataset()]}
LOSS_TYPES = ["triplet", "arcface"]

 
PROJECTION_FILE_RE = re.compile(r"^(?P<dim>\d+)d_margin(?P<margin>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
CHECKPOINT_VAL_MAP_RE = re.compile(r"val_mAP=(?P<val_map>\d+\.\d+)")


# --------------------------------------------------------------------------
# Odkrywanie dostępnych konfiguracji na dysku
# --------------------------------------------------------------------------

def discover_margin_seed_files(loss_type: str, model_name: str, dataset_name: str) -> dict:
    """Skanuje `metric_learning/projection_layers/{loss_type}/{model}/{dataset}/`
    i zwraca {margin: {seed: projection_path}} na podstawie nazw plików."""
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
    """Odczytuje val_mAP zakodowany w nazwie checkpointu Lightning
    (np. best-epoch=81-val_mAP=0.5075.ckpt), bez wczytywania samego pliku."""
    checkpoints_dir = LOGS_ROOT / loss_type / model_name / dataset_name / f"margin{margin}_seed{seed}" / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None
    for ckpt_file in checkpoints_dir.glob("best-*.ckpt"):
        match = CHECKPOINT_VAL_MAP_RE.search(ckpt_file.name)
        if match:
            return float(match.group("val_map"))
    return None


def find_lightning_checkpoint(loss_type: str, model_name: str, dataset_name: str, margin: float, seed: int) -> Path:
    checkpoints_dir = LOGS_ROOT / loss_type / model_name / dataset_name / f"margin{margin}_seed{seed}" / "checkpoints"
    ckpt_files = sorted(checkpoints_dir.glob("best-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"Brak checkpointu Lightning w {checkpoints_dir}")
    if len(ckpt_files) > 1:
        print(f"    [uwaga] więcej niż jeden checkpoint w {checkpoints_dir}, biorę: {ckpt_files[0].name}")
    return ckpt_files[0]


def select_best_margin(loss_type: str, model_name: str, dataset_name: str, margin_to_seed_paths: dict) -> float:
    """Margines uznawany za najlepszy to ten, dla którego zapisano komplet
    finałowych seedów (FINAL_SEEDS) - dokładnie tak trenował oryginalny
    pipeline (tylko wybrany margines dostawał wszystkie 5 seedów). Jeśli
    więcej niż jeden margines ma komplet seedów, wybierany jest ten
    z najwyższym średnim val_mAP odczytanym z nazw checkpointów."""
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


# --------------------------------------------------------------------------
# Budowa modelu do ewaluacji - dwa tryby wczytywania
# --------------------------------------------------------------------------

def load_wrapper_from_lightning_checkpoint(wrapper: nn.Module, ckpt_path: Path) -> nn.Module:
    """Wczytuje CAŁY model (backbone + head) z checkpointu Lightning zamiast
    tylko osobno zapisanej warstwy projekcji. Przydatne, gdy backbone też był
    fine-tune'owany podczas treningu, a nie tylko zamrożony."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state_dict = checkpoint["state_dict"]

    # Lightning zapisuje wagi z prefiksem nazwy atrybutu modułu (typowo
    # "model." dla `self.model = wrapper` w ProjectionLayerModule). Jeśli u
    # Ciebie atrybut nazywa się inaczej, dopisz go do listy poniżej.
    for prefix in ("model.", "wrapper.", ""):
        stripped = {
            key[len(prefix):]: value
            for key, value in full_state_dict.items()
            if key.startswith(prefix)
        }
        if not stripped:
            continue
        missing, unexpected = wrapper.load_state_dict(stripped, strict=False)
        if not missing:
            if unexpected:
                print(f"    [uwaga] nieużyte klucze z checkpointu (prefiks {prefix!r}): {unexpected}")
            return wrapper

    raise RuntimeError(
        f"Nie udało się dopasować state_dict z {ckpt_path} do ProjectionLayerWrapper "
        "- sprawdź nazwę atrybutu modelu w ProjectionLayerModule i dopisz właściwy prefiks."
    )


def build_eval_model(backbone, model_schema, loss_type: str, dataset_name: str, margin: float, seed: int,
                      projection_path: Path, load_mode: str) -> nn.Module:
    head = nn.Linear(in_features=model_schema.out_features, out_features=PROJECTION_DIM)

    if load_mode == "head_only":
        # Zamrożony, pretrenowany backbone + osobno zapisana warstwa projekcji.
        head.load_state_dict(torch.load(projection_path, map_location="cpu"))
        eval_model = nn.Sequential(backbone, L2NormLayer(dim=1), head, L2NormLayer(dim=1))
    elif load_mode == "full_checkpoint":
        # Cały model (backbone + head) odtworzony z checkpointu Lightning.
        ckpt_path = find_lightning_checkpoint(loss_type, model_schema.name, dataset_name, margin, seed)
        wrapper = ProjectionLayerWrapper(backbone=backbone, head=head)
        load_wrapper_from_lightning_checkpoint(wrapper, ckpt_path)
        eval_model = nn.Sequential(wrapper, L2NormLayer(dim=1))
    else:
        raise ValueError(f"Nieznany load_mode: {load_mode}")

    eval_model.requires_grad_(False)
    eval_model.eval()
    return eval_model


# --------------------------------------------------------------------------
# Ewaluacja na zbiorze testowym
# --------------------------------------------------------------------------

def evaluate_on_test(model_schema, dataset_schema, loss_type: str, margin: float, seed: int,
                      projection_path: Path, load_mode: str):
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

    eval_model = build_eval_model(
        backbone, model_schema, loss_type, dataset_schema.name, margin, seed, projection_path, load_mode
    )
    module = EvaluationModule(model=eval_model)

    test_metrics = MetricCollection({"Recall@1": RetrievalHitRate(top_k=1), "mAP": RetrievalMAP()})

    base_gallery_path = path_creator.create_path(
        model_schema.name, dataset_schema.name, output_dim=PROJECTION_DIM,
        attack_name=None, margin=margin, seed=seed, iter_count=None,
    )
    # Seed jawnie dopisany na końcu ścieżki, żeby różne seedy nie nadpisywały
    # sobie wzajemnie cache'u/features.
    gallery_path = f"features/{loss_type}/{base_gallery_path}"
    print(gallery_path)
    cache_path = f"cache/{gallery_path}"
    last_cached_file_idx = cache_tools.get_last_cached_batch_idx(cache_path)
    start_idx = last_cached_file_idx + 1 if last_cached_file_idx != -1 else 0

    trainer = Trainer(callbacks=[LeaveOneOutMetricsCallback(test_metrics, gallery_path, cache_path, start_idx)])
    dataloader = itertools.islice(datamodule.test_dataloader(), start_idx, None)
    return trainer.test(model=module, dataloaders=[dataloader])


def _aggregate_test_metrics(seed_results: list) -> dict:
    metric_values: dict = {}
    for r in seed_results:
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


def evaluate_configuration(model_schema, dataset_schema, loss_type: str, load_mode: str):
    margin_to_seed_paths = discover_margin_seed_files(loss_type, model_schema.name, dataset_schema.name)
    if not margin_to_seed_paths:
        directory = PROJECTION_LAYERS_ROOT / loss_type / model_schema.name / dataset_schema.name
        print(f"  Brak zapisanych warstw projekcji w {directory} - pomijam.")
        return None

    try:
        best_margin = select_best_margin(loss_type, model_schema.name, dataset_schema.name, margin_to_seed_paths)
    except ValueError as exc:
        print(f"  {exc}")
        return None

    seed_paths = margin_to_seed_paths[best_margin]
    seeds_to_eval = sorted(seed_paths.keys())
    print(f"  Najlepszy margines: {best_margin} (seedy: {seeds_to_eval})")

    seed_results = []
    for seed in seeds_to_eval:
        val_map = read_val_map_from_checkpoint(loss_type, model_schema.name, dataset_schema.name, best_margin, seed)
        test_metrics = evaluate_on_test(
            model_schema, dataset_schema, loss_type, best_margin, seed, seed_paths[seed], load_mode
        )
        seed_results.append({
            "seed": seed,
            "val_mAP": val_map,
            "projection_path": str(seed_paths[seed]),
            "test_metrics": test_metrics,
        })

    aggregated_test = _aggregate_test_metrics(seed_results)
    for metric_name, stats in aggregated_test.items():
        print(f"  {metric_name} = {stats['mean']:.4f} +/- {stats['std']:.4f}")

    return {
        "best_margin": best_margin,
        "seed_results": seed_results,
        "test_metrics_aggregated": aggregated_test,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ewaluacja wcześniej wytrenowanych modeli metric learning.")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_SCHEMAS.keys()), choices=list(MODEL_SCHEMAS.keys()),
        help="Które modele ewaluować (domyślnie wszystkie).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_SCHEMAS.keys()), choices=list(DATASET_SCHEMAS.keys()),
        help="Które zbiory danych ewaluować (domyślnie wszystkie).",
    )
    parser.add_argument(
        "--loss-types", nargs="+", default=LOSS_TYPES, choices=LOSS_TYPES,
        help="Które funkcje straty ewaluować (domyślnie obie).",
    )
    parser.add_argument(
        "--load-mode", choices=["head_only", "full_checkpoint"], default="head_only",
        help="'head_only': zamrożony backbone + zapisana warstwa projekcji (domyślne, szybsze). "
             "'full_checkpoint': cały model (backbone+head) wczytany z checkpointu Lightning.",
    )
    parser.add_argument(
        "--output-path", type=str, default="metric_learning_eval_results.pt",
        help="Gdzie zapisać wyniki ewaluacji.",
    )
    args = parser.parse_args()

    all_results = {}
    for model_name in args.models:
        model_schema = MODEL_SCHEMAS[model_name]
        for dataset_name in args.datasets:
            dataset_schema = DATASET_SCHEMAS[dataset_name]
            for loss_type in args.loss_types:
                print(f"\n===== EWALUACJA: {model_name} / {dataset_name} / {loss_type} =====")
                result = evaluate_configuration(model_schema, dataset_schema, loss_type, args.load_mode)
                if result is not None:
                    all_results[(model_name, dataset_name, loss_type)] = result

    torch.save(all_results, args.output_path)
    print(f"\nZapisano wyniki ewaluacji do: {Path(args.output_path).resolve()}")