import torch

_EPSILON = 1e-10  # small number to prevent indeterminate division


def normalize_batch(advantages_batch, normalize: bool = True):
    if normalize:
        normalized_advantages_batch = (advantages_batch - advantages_batch.mean(dim=(1, 2), keepdim=True)) / (
                advantages_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
    else:
        normalized_advantages_batch = advantages_batch
    return normalized_advantages_batch
