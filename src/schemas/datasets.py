class CUBDataset:
  name = 'cub'
  path = '/data/cybersecurity/final/data/cub'
  train_classes = list(range(0, 70))
  val_classes = list(range(70, 100))
  test_classes = list(range(100, 200))

class CarsDataset:
  name = 'cars'
  path = '/data/cybersecurity/final/data/cars'
  train_classes = list(range(0, 70))
  val_classes = list(range(70, 98))
  test_classes = list(range(98, 196))

class SOPDataset:
  name = 'sop'
  path = '/data/cybersecurity/final/data/sop'
  train_classes = list(range(0, 10318))
  val_classes = list(range(10318, 11318))
  test_classes = list(range(11318, 22634))