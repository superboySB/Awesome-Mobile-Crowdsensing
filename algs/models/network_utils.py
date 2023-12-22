import numpy as np
import torch


def layer_init(layer: torch.nn.Module, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal Initialization of NN.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
