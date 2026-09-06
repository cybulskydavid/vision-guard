import functools
import torch.nn as nn

def ensure_eval(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = next((arg for arg in args if isinstance(arg, nn.Module)), None)
        if not model:
            model = next((kwarg for kwarg in kwargs.values() if isinstance(kwarg, nn.Module)), None)
        if model is None:
            raise ValueError("@ensure_eval: No PyTorch model (nn.Module) found in arguments.")

        is_training = model.training
        model.eval()
        
        try:
            return func(*args, **kwargs)
        finally:
            model.train(is_training)
            
    return wrapper