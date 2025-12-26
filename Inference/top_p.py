import torch
import torch.nn as nn


def sample_top_p(probs,p):
    """
    Args
    
    :param probs: probabilties of the tokens
    :param p: standard parameters to choose how much cumilitive probability of the token we need to store
    """
    probs_sort, porbs_idx = torch.sort(probs, dim=-1,descending=True)
    probs_sum = torch.cumsum(probs_sort,dim=-1)
    mask = probs_sum - probs_sort>p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1,keepdim=True))
    next_token = torch.multinomial(probs_sort,num_samples=1)
    next_token = torch.gather(porbs_idx,-1,next_token)
    return next_token
