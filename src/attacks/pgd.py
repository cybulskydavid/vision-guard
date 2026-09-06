import torch
import torch.nn.functional as F

from src.attacks.base import BaseAttack
from src.utils.decorators import ensure_eval


class PGDAttack(BaseAttack):
    def __init__(
        self, 
        epsilon: float, 
        alpha: float, 
        iter_count: int,
        num_restarts: int,
        mining_function,
        gallery,
        labels,
        n_positives: int,
        n_negatives: int,
        **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.iter_count = iter_count
        self.num_restarts = num_restarts
        self.mining_function = mining_function
        self.gallery = gallery
        self.labels = labels
        self.n_positives = n_positives
        self.n_negatives = n_negatives

    @ensure_eval
    @torch.set_grad_enabled(True)
    @torch.inference_mode(False)
    def execute(self, model, input, image_idx, device):
        batch_size = input.shape[0]

        input = input.clone().detach()

        with torch.no_grad():
            pos_samples, neg_samples = self.mining_function(
                self.gallery, self.labels, image_idx, 
                self.n_positives, self.n_negatives, device
            )
            pos_samples = F.normalize(pos_samples, p=2, dim=-1)
            neg_samples = F.normalize(neg_samples, p=2, dim=-1)
        
        best_delta = torch.zeros_like(input, device=device)
        best_adv_loss = torch.ones(batch_size, device=device) * float('inf')

        for _ in range(self.num_restarts):
            delta = torch.zeros_like(input, device=device)
            delta.uniform_(-self.epsilon, self.epsilon)
            delta = torch.clamp(input + delta, 0.0, 1.0) - input
            delta.requires_grad_(True)

            for _ in range(self.iter_count):
                adv_input = input + delta
                
                adv_features = model(adv_input)
                adv_features = F.normalize(adv_features, p=2, dim=-1).unsqueeze(1)

                sim_pos = F.cosine_similarity(adv_features, pos_samples, dim=-1).mean(dim=1)
                sim_neg = F.cosine_similarity(adv_features, neg_samples, dim=-1).mean(dim=1)

                loss = (sim_pos - sim_neg).sum()

                model.zero_grad()
                if delta.grad is not None: 
                    delta.grad.zero_()

                loss.backward()

                with torch.no_grad():
                    if delta.grad is not None: 
                        delta.add_(-self.alpha * delta.grad.sign())
                    
                    new_delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    new_delta = torch.clamp(input + new_delta, 0.0, 1.0) - input
                    delta.copy_(new_delta)

            with torch.no_grad():
                final_adv_input = torch.clamp(input + delta.detach(), 0.0, 1.0)
                final_adv_features = model(final_adv_input)
                final_adv_features = F.normalize(final_adv_features, p=2, dim=-1).unsqueeze(1)

                f_sim_pos = F.cosine_similarity(final_adv_features, pos_samples, dim=-1).mean(dim=1)
                f_sim_neg = F.cosine_similarity(final_adv_features, neg_samples, dim=-1).mean(dim=1)
                
                final_loss = f_sim_pos - f_sim_neg

                idx_improved = final_loss < best_adv_loss
                best_adv_loss[idx_improved] = final_loss[idx_improved]
                best_delta[idx_improved] = delta.detach()[idx_improved]

        final_adv_input = torch.clamp(input + best_delta, 0.0, 1.0)
        return final_adv_input.detach()
    
    @property
    def margin(self):
      return self.epsilon