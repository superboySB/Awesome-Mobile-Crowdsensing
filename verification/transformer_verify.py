import torch.nn as nn
import torch

torch.manual_seed(0)
src = tgt = torch.rand((5, 32, 2))
transformer_model = nn.Transformer(nhead=1, num_encoder_layers=2, num_decoder_layers=2,
                                   dim_feedforward=64, d_model=2)
out = transformer_model(src, tgt)
print(out)
