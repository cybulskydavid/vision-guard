import torch
import torch.nn.functional as F

from src.attacks.base import BaseAttack
from src.utils.decorators import ensure_eval


class CWAttack(BaseAttack):
  def __init__(self, 
               kappa: float, 
               iter_count: int, 
               learning_rate: float,
              #  loss_function,
               mining_function,
               gallery,
               labels,
               n_positives,
               n_negatives,
               binary_search_steps: int = 7,
               init_const: float = 10,
               eps: float = 1e-6,
    ):
    super().__init__()
    self.kappa = kappa
    self.iter_count = iter_count
    self.learning_rate = learning_rate
    self.optimizer_cls = torch.optim.Adam
    # self.loss_function = loss_function
    self.mining_function = mining_function
    self.gallery = gallery
    self.labels = labels
    self.n_positives = n_positives
    self.n_negatives = n_negatives
    self.binary_search_steps = binary_search_steps
    self.init_const = init_const
    self.eps = eps


  @ensure_eval
  @torch.set_grad_enabled(True)
  @torch.inference_mode(False)
  def execute(self, model, images, image_idx, device):
    batch_size = images.size(0)
    images = images.clone().detach()

    with torch.no_grad():
      pos_samples, neg_samples = self.mining_function(
          self.gallery, self.labels, image_idx, 
          self.n_positives, self.n_negatives, device
      )
      
      pos_samples = F.normalize(pos_samples, p=2, dim=-1)
      neg_samples = F.normalize(neg_samples, p=2, dim=-1)

    input_clipped = torch.clamp(images, self.eps, 1.0 - self.eps)
    w_init = torch.atanh(input_clipped * 2 - 1)

    lower_bound = torch.zeros(batch_size, device=device)
    const = torch.ones(batch_size, device=device) * self.init_const
    upper_bound = torch.ones(batch_size, device=device) * 1e6

    best_l2 = torch.full((batch_size,), float('inf'), device=device)
    best_adv_input = images.clone().detach()

    best_adv_loss_overall = torch.full((batch_size,), float('inf'), device=device)
    best_effort_input = images.clone().detach()

    for _ in range(self.binary_search_steps):
      w = w_init.clone().detach().requires_grad_(True)
      success = torch.zeros(batch_size, dtype=torch.bool, device=device)
      optimizer = self.optimizer_cls([w], lr=self.learning_rate)

      for _ in range(self.iter_count):
        adv_input = 0.5 * (torch.tanh(w) + 1.0)
        l2_loss = torch.sum((adv_input - images)**2, dim=(1, 2, 3))

        adv_features = F.normalize(model(adv_input), p=2, dim=-1).unsqueeze(1)

        sim_pos = F.cosine_similarity(adv_features, pos_samples, dim=-1)
        sim_neg = F.cosine_similarity(adv_features, neg_samples, dim=-1)

        sim_pos_exp = sim_pos.unsqueeze(2) 
        sim_neg_exp = sim_neg.unsqueeze(1) 
        
        triplet_matrix = sim_pos_exp - sim_neg_exp + self.kappa
        
        triplet_losses = F.relu(triplet_matrix).view(batch_size, -1)

        adv_loss = triplet_losses.mean(dim=1)

        loss = (l2_loss + const * adv_loss).sum()

        optimizer.zero_grad()
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
          is_successful = (adv_loss <= 1e-4)
          success |= is_successful

          improved_l2 = is_successful & (l2_loss < best_l2)
          best_l2[improved_l2] = l2_loss.detach()[improved_l2]
          best_adv_input[improved_l2] = adv_input.detach()[improved_l2]

          improved_loss = adv_loss < best_adv_loss_overall
          best_adv_loss_overall[improved_loss] = adv_loss.detach()[improved_loss]
          best_effort_input[improved_loss] = adv_input.detach()[improved_loss]
      
      with torch.no_grad():
        fail = ~success
        upper_bound[success] = torch.minimum(upper_bound[success], const[success])
        lower_bound[fail] = torch.maximum(lower_bound[fail], const[fail])

        bounded = upper_bound < 1e5
        calc_mean_mask = success | (fail & bounded)
        mult_mask = fail & ~bounded
        
        const[calc_mean_mask] = (lower_bound[calc_mean_mask] + upper_bound[calc_mean_mask]) / 2.0
        const[mult_mask] *= 5.0

    failed_completely = (best_l2 == float('inf'))
    best_adv_input[failed_completely] = best_effort_input[failed_completely]

    return torch.clamp(best_adv_input, 0.0, 1.0)

  @property
  def margin(self):
    return self.kappa
