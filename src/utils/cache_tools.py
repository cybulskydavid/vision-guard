from pathlib import Path


def get_last_cached_batch_idx(cache_path):
  path = Path(cache_path)

  if not path.is_dir():
    return -1
  
  indices = []
  for file_path in path.glob('*.pt'):
    idx_str = ''.join(filter(str.isdigit, file_path.stem))

    if idx_str:
      indices.append(int(idx_str))

  return max(indices) if indices else -1