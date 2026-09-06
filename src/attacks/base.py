from abc import ABC, abstractmethod


class BaseAttack(ABC):
  def __call__(self, model, images, images_idx, device, *args, **kwds):
    return self.execute(model, images, images_idx, device)

  @abstractmethod
  def execute(self, model, images, images_idx, device):
    pass

  @property
  @abstractmethod
  def margin(self):
    pass