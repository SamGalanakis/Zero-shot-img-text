import torch
import einops
from torch import nn
from.nets import MLP

class TextEmbdedder(nn.Module):
    def __init__(self,input_dim,hidden_dims,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = MLP(input_dim,hidden_dims,emb_dim)

    def forward(self,x):
        # Could try adding the vis features to this linear layer
        x = self.linear(x)
        x = einops.reduce(x,'n b e -> b e','max')
        return x
