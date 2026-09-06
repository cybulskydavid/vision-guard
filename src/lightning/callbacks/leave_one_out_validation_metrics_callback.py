import torch
from pytorch_lightning import Callback
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMAP


class LeaveOneOutValidationMetricsCallback(Callback):
    def __init__(self, top_k: int = 1):
        super().__init__()
        self.top_k = top_k
        self._embeddings = []
        self._labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._embeddings.clear()
        self._labels.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._embeddings.append(outputs["embeddings"].detach().cpu())
        self._labels.append(outputs["labels"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._embeddings:
            return

        device = pl_module.device

        embeddings = torch.cat(self._embeddings, dim=0).to(device)
        labels = torch.cat(self._labels, dim=0).to(device)
        n = embeddings.shape[0]

        sim_matrix = embeddings @ embeddings.T
        sim_matrix.fill_diagonal_(float("-inf"))
        target = labels.unsqueeze(1) == labels.unsqueeze(0)
        indexes = torch.arange(n).unsqueeze(1).expand(n, n)

        val_map = RetrievalMAP().to(device)(sim_matrix, target, indexes=indexes)
        val_r1 = RetrievalHitRate(top_k=self.top_k).to(device)(sim_matrix, target, indexes=indexes)

        pl_module.log("val_mAP", val_map, prog_bar=True)
        pl_module.log("val_R1", val_r1, prog_bar=True)

        self._embeddings.clear()
        self._labels.clear()