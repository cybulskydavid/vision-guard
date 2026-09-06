import os
from typing import Any, Callable, cast

import torch
from torch.utils.data import Dataset


class ClassSubset(Dataset):
    def __init__(
            self, 
            dataset: Dataset, 
            allowed_classes: list[int],
            transform: Callable[[Any], torch.Tensor] | None = None
        ) -> None:
        self.dataset = dataset
        self.transform = transform

        self.class_map: dict[int, int] = {
            old_cls: new_idx for new_idx, old_cls in enumerate(sorted(allowed_classes))
        }

        allowed_set = set(allowed_classes)
        self.valid_indices: list[int] = [
            i for i, label in enumerate(self._get_targets(dataset))
            if label in allowed_set
        ]


    @staticmethod
    def _get_targets(dataset: Dataset) -> list[int]:
        targets = getattr(dataset, 'targets', None)
        if targets is not None:
            return cast(list[int], targets)
        samples = getattr(dataset, 'samples', None)
        if samples is not None:
            return [sample[1] for sample in samples]

        raise AttributeError("targets or samples attribute not found in the dataset.")

    def _get_filename(self, real_idx: int) -> str:
        samples = getattr(self.dataset, 'samples', None)
        if samples is not None:
            path = samples[real_idx][0]
            return os.path.basename(path)

        return f"unknown_image_{real_idx}"

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        real_idx = self.valid_indices[idx]
        data, old_label = self.dataset[real_idx]
        data = self.transform(data) if self.transform else data
        filename = self._get_filename(real_idx)
        return data, self.class_map[old_label], idx, filename
        # return data, self.class_map[old_label]