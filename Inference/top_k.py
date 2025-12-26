import torch

def sample_top_k(probs, k):
    """
    Args:
        probs: Tensor of shape (..., vocab_size)
        k: number of top tokens to keep
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sort[..., k:] = 0.0
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
