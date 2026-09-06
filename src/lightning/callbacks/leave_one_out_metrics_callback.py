import os
from pathlib import Path

from pytorch_lightning import Callback
import torch
from torchmetrics import MetricCollection
from src.utils.logger import setup_logger
import torch.nn.functional as F

class LeaveOneOutMetricsCallback(Callback):
  def __init__(self, test_metrics: MetricCollection, save_path: str, cache_path: str, batch_offset: int):
    super().__init__()
    self.embeddings = []
    self.labels = []
    self.images_idx = []
    self.images_name = []
    self.adv_images = []

    self.metrics = { 'test': test_metrics }
    self.logger = setup_logger(__name__)
    self.save_path = f'{save_path}'
    self.cache_path = cache_path
    self.batch_offset = batch_offset

    if self.cache_path:
      os.makedirs(self.cache_path, exist_ok=True)


  def _evaluate_chunked(self, stage, device, chunk_size=4096):
    if not self.embeddings:
      return
    
    metrics = self.metrics[stage].to(device)
    embeddings = F.normalize(torch.cat(self.embeddings).to(device))
    labels = torch.cat(self.labels).to(device)

    n_embeddings = len(embeddings)

    accumulated_results = {}

    for start_idx in range(0, n_embeddings, chunk_size):
      emb_chunk = embeddings[start_idx : start_idx + chunk_size]
      lab_chunk = labels[start_idx : start_idx + chunk_size]

      n_chunk = emb_chunk.size(0)

      preds = torch.mm(emb_chunk, embeddings.t())
      targets = (lab_chunk.unsqueeze(1) == labels.unsqueeze(0))
      indexes = torch.arange(start_idx, start_idx + n_chunk, device=device).view(-1,1).expand(n_chunk, n_embeddings)

      keep_mask = torch.ones_like(preds, dtype=torch.bool)
      row_indices = torch.arange(n_chunk, device=device)
      col_indices = torch.arange(start_idx, start_idx + n_chunk, device=device)
      keep_mask[row_indices, col_indices] = False

      metrics.update(preds=preds[keep_mask], target=targets[keep_mask], indexes=indexes[keep_mask])

      chunk_results = metrics.compute()
      
      for metric_name, score in chunk_results.items():
        if metric_name not in accumulated_results:
          accumulated_results[metric_name] = 0.0
        accumulated_results[metric_name] += score.item() * n_chunk

      metrics.reset()

      del preds, targets, indexes, keep_mask
      if device.type == 'cuda':
        torch.cuda.empty_cache()

    final_results = {name: total / n_embeddings for name, total in accumulated_results.items()}
    return final_results


  def _log_metrics(self, stage, results):
    for name, value in results.items():
      metric_name = f"{stage}/{name}"
      messagge = f'{metric_name} : {value}'
      self.logger.info(messagge)
  

  def _reset_state(self):
    self.embeddings.clear()
    self.labels.clear()
    self.images_idx.clear()
    self.images_name.clear()
    self.adv_images.clear()
    self.metrics['test'].reset()

  
  def on_test_epoch_start(self, trainer, pl_module):
    self._reset_state()


  def on_test_epoch_end(self, trainer, pl_module):
    folder_path = Path(self.cache_path)
    cache_files = sorted(folder_path.glob("batch_*.pt"), key=lambda x: int(x.stem.split('_')[1]))

    for path in cache_files:
      if path.is_file():
        cache_data = torch.load(path)
        self.embeddings.append(cache_data['embeddings'])
        self.labels.append(cache_data['labels'])
        if cache_data['images_idx'] is not None:
          self.images_idx.append(cache_data['images_idx'])
        if cache_data['adv_images'] is not None:
          self.adv_images.append(cache_data['adv_images'])
        self.images_name.extend(cache_data.get('images_name', []))

    if not self.embeddings:
      return
    
    results = self._evaluate_chunked('test', pl_module.device)
    self._log_metrics('test', results)

    all_embeddings = torch.cat(self.embeddings)
    all_labels = torch.cat(self.labels)
    all_images_idx = torch.cat(self.images_idx)
    adv_images = torch.cat(self.adv_images)

    data_to_save = {
              'embeddings': all_embeddings,
              'labels': all_labels,
              'images_idx': all_images_idx,
              'images_name': self.images_name
          }
    
    os.makedirs(os.path.dirname(f'{self.save_path}/features.pt'), exist_ok=True)
    torch.save(data_to_save, f'{self.save_path}/features.pt')
    self.logger.info(f"Galeria zapisana pomyślnie pod adresem: {self.save_path}/features.pt")
    
    if cache_data['adv_images'] is not None:
      images_to_save = {
              'adv_images': adv_images,
              'images_idx': all_images_idx
      }

      os.makedirs(os.path.dirname(f'{self.save_path}/images.pt'), exist_ok=True)
      torch.save(images_to_save, f'{self.save_path}/images.pt')
      self.logger.info(f"Obrazy zapisano pomyślnie pod adresem: {self.save_path}/images.pt")

    for file in Path(self.cache_path).glob("batch_*.pt"):
      try:
        file.unlink()
      except Exception as e:
        self.logger.warning(f"Nie udało się usunąć pliku cache {file}: {e}")
    self.logger.info("Pliki cache zostały wyczyszczone.")


  def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
    features = outputs['features'].detach().cpu()
    labels = outputs['labels'].detach().cpu()
    img_idx = outputs['images_idx'].detach().cpu()
    img_names = outputs.get('images_name', [])
    adv_img_raw = outputs.get('adv_images')

    if adv_img_raw is not None:
      adv_images = adv_img_raw.detach().cpu()
    else:
      adv_images = torch.tensor([])

    real_batch_idx = batch_idx + self.batch_offset
    cache_file = os.path.join(self.cache_path, f'batch_{real_batch_idx}.pt')

    torch.save({
      'embeddings': features,
      'labels': labels,
      'images_idx': img_idx,
      'images_name': img_names,
      'adv_images': adv_images
    }, cache_file)