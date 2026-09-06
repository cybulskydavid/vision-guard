import clip
import torch
import torchvision.transforms as transforms
from transformers import AutoModel
from torch import nn

class TensorOutputModel(nn.Module):
  def __init__(self, model, *args, **kwargs):
    super().__init__()
    self.model = model

  def forward(self, *args, **kwargs):
    outputs = self.model(*args, **kwargs)
    return outputs.last_hidden_state[:, 0, :]
     

def load_model_and_transforms(model_name, device):
  imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
  clip_mean, clip_std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
  
  def build_unified_transform(mean, std):
    return transforms.Compose([
      transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor()
    ]), transforms.Compose([transforms.Normalize(mean=mean, std=std)])

  if model_name == 'clip_vitb':
      full_model, _ = clip.load("ViT-B/16", device=device)
      model = full_model.visual.float()
      transform_image, transform_tensor = build_unified_transform(clip_mean, clip_std)
  elif model_name == 'clip_resnet50':
      full_model, _ = clip.load("RN50", device=device)
      model = full_model.visual.float()
      transform_image, transform_tensor = build_unified_transform(clip_mean, clip_std)
  elif model_name == 'dino_vitb':
      model = TensorOutputModel(AutoModel.from_pretrained("facebook/dino-vitb16", use_safetensors=True).float())
      transform_image, transform_tensor = build_unified_transform(imagenet_mean, imagenet_std)
  elif model_name == 'dino_vits':
      model = TensorOutputModel(AutoModel.from_pretrained("facebook/dino-vits16", use_safetensors=True).float())
      transform_image, transform_tensor = build_unified_transform(imagenet_mean, imagenet_std)
  elif model_name == 'dino_resnet50':
      model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').float()
      transform_image, transform_tensor = build_unified_transform(imagenet_mean, imagenet_std)
  else:
      raise ValueError(f"Unknown model: {model_name}")

  model = model.to(device)
  model.eval()
  return model, transform_image, transform_tensor