import os
from pathlib import Path

import torch
from torchmetrics import MetricCollection
from src.lightning.callbacks.leave_one_out_metrics_callback import LeaveOneOutMetricsCallback
import torch.nn.functional as F

class AdvLeaveOneOutMetricsCallback(LeaveOneOutMetricsCallback):
    def __init__(self, save_path: str, cache_path: str, gallery_embeddings: torch.Tensor, gallery_labels: torch.Tensor, gallery_idx: torch.Tensor, test_metrics: MetricCollection, batch_offset: int):
        super().__init__(test_metrics=test_metrics, 
            save_path=save_path, 
            cache_path=cache_path, 
            batch_offset=batch_offset
        )
        self.gallery_embeddings = gallery_embeddings.detach().cpu()
        self.gallery_labels = gallery_labels.detach().cpu()
        self.gallery_idx = gallery_idx.detach().cpu()

    def _evaluate_chunked(self, stage, device, chunk_size=4096):
        if not self.embeddings:
            return {}
        
        metrics = self.metrics[stage].to(device)
        metrics.reset()

        query_embeddings = F.normalize(torch.cat(self.embeddings).to(device))
        query_labels = torch.cat(self.labels).to(device)
        query_img_idx = torch.cat(self.images_idx).to(device)

        gallery_embeddings = F.normalize(self.gallery_embeddings.to(device))
        gallery_labels = self.gallery_labels.to(device)
        gallery_idx_device = self.gallery_idx.to(device)

        n_queries = len(query_embeddings)
        n_gallery = len(gallery_embeddings)

        accumulated_results = {}

        for start_idx in range(0, n_queries, chunk_size):
            emb_chunk = query_embeddings[start_idx : start_idx + chunk_size]
            lab_chunk = query_labels[start_idx : start_idx + chunk_size]
            idx_chunk = query_img_idx[start_idx : start_idx + chunk_size]

            n_chunk = emb_chunk.size(0)

            preds = torch.mm(emb_chunk, gallery_embeddings.t())
            targets = (lab_chunk.unsqueeze(1) == gallery_labels.unsqueeze(0))
            
            indexes = torch.arange(start_idx, start_idx + n_chunk, device=device).view(-1, 1).expand(n_chunk, n_gallery)

            keep_mask = (idx_chunk.unsqueeze(1) != gallery_idx_device.unsqueeze(0))
            row_indices = torch.arange(n_chunk, device=device)
            
            keep_mask[row_indices, idx_chunk] = False

            metrics.update(
                preds=preds[keep_mask],
                target=targets[keep_mask],
                indexes=indexes[keep_mask]
            )

            chunk_results = metrics.compute()
            
            for metric_name, score in chunk_results.items():
                if metric_name not in accumulated_results:
                    accumulated_results[metric_name] = 0.0
                accumulated_results[metric_name] += score.item() * n_chunk

            metrics.reset()

            del preds, targets, indexes, keep_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        final_results = {name: total / n_queries for name, total in accumulated_results.items()}
        return final_results