import torch

def positives_negatives_miner(gallery, labels, indexes, n_positive, n_negative, device):
  with torch.no_grad():
    if not isinstance(indexes, torch.Tensor):
        indexes = torch.tensor(indexes)

    B = indexes.size(0)

    indexes_gpu = indexes.to(device=device)
    gallery_gpu = gallery.to(device=device)
    labels_gpu = labels.to(device=device)

    queries = gallery_gpu[indexes_gpu]
    dist_mat = torch.cdist(queries, gallery_gpu, p=2.0)
    del queries

    query_labels = labels_gpu[indexes_gpu]
    pos_mask = query_labels.unsqueeze(1) == labels_gpu.unsqueeze(0)
    neg_mask = ~pos_mask
    del query_labels, labels_gpu

    batch_idx = torch.arange(B, device=device)
    dist_mat[batch_idx, indexes_gpu] = float('inf')
    del batch_idx, indexes_gpu

    pos_dist_mat = dist_mat.clone() 
    pos_dist_mat[neg_mask] = float('inf')
    del neg_mask  
    
    _, pos_indices = torch.topk(pos_dist_mat, k=n_positive, dim=1, largest=False)
    del pos_dist_mat 

    dist_mat[pos_mask] = float('inf')
    del pos_mask
    
    _, neg_indices = torch.topk(dist_mat, k=n_negative, dim=1, largest=False)
    del dist_mat

  pos_vectors = gallery_gpu[pos_indices]
  neg_vectors = gallery_gpu[neg_indices]
  
  return pos_vectors, neg_vectors