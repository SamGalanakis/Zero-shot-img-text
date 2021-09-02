import torch
import einops
from torch import nn
from.nets import MLP

class MatchPredictor(nn.Module):
    def __init__(self,emb_dim_visual,emb_dim_text,hidden_dims):
        super().__init__()

        self.mlp = MLP(emb_dim_visual+emb_dim_text, hidden_dims,1,nonlin= torch.nn.GELU())

    def forward(self,visual,text):
        visual = einops.repeat(visual,'b e -> b t e',t = text.shape[0])
        text = einops.repeat(text,'t e -> b t e',b = visual.shape[0])
        x = torch.cat((visual,text),dim=-1)
        match_val = self.mlp(x).squeeze().softmax(dim=-1)
        return match_val
