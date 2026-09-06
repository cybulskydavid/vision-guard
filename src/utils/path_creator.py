def create_path(model_name, dataset_name, output_dim, attack_name, iter_count, margin, seed):
  if attack_name:
    return f'features/{attack_name}{f'/{iter_count}/' if iter_count else ''}/{margin:4f}/{model_name}/{dataset_name}/{output_dim}/{seed}'
  elif seed:
    return f'features/clean/{model_name}/{dataset_name}/{output_dim}/{seed}'
  else:
    return f'features/clean/{model_name}/{dataset_name}/{output_dim}'