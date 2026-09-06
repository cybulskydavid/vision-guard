import torch
import torch.nn.functional as F

from src.attacks.base import BaseAttack
from src.utils.decorators import ensure_eval

class FGSMAttack(BaseAttack):
    def __init__(
        self, 
        epsilon: float, 
        mining_function,
        gallery,
        labels,
        n_positives: int,
        n_negatives: int,
        **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.mining_function = mining_function
        self.gallery = gallery.cpu()
        self.labels = labels.cpu()
        self.n_positives = n_positives
        self.n_negatives = n_negatives

    @ensure_eval
    @torch.set_grad_enabled(True)
    @torch.inference_mode(False)
    def execute(self, model, input, image_idx, device):
        input = input.clone().detach()
        input.requires_grad_(True)

        with torch.no_grad():
            pos_samples, neg_samples = self.mining_function(
                self.gallery, self.labels, image_idx, 
                self.n_positives, self.n_negatives, device
            )
            pos_samples = F.normalize(pos_samples, p=2, dim=-1)
            neg_samples = F.normalize(neg_samples, p=2, dim=-1)

        adv_features = model(input)
        adv_features = F.normalize(adv_features, p=2, dim=-1).unsqueeze(1)

        sim_pos = F.cosine_similarity(adv_features, pos_samples, dim=-1).mean(dim=1)
        sim_neg = F.cosine_similarity(adv_features, neg_samples, dim=-1).mean(dim=1)

        loss = (sim_pos - sim_neg).sum()

        model.zero_grad()
        if input.grad is not None:
            input.grad.zero_()
            
        loss.backward()

        with torch.no_grad():
            adv_input = input - self.epsilon * input.grad.sign()
            adv_input = torch.clamp(adv_input, 0.0, 1.0)

        return adv_input.detach()
      
    @property
    def margin(self):
      return self.epsilon