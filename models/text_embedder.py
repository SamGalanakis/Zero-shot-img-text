import torch
import einops
from torch import nn
from.nets import MLP

class MatchPredictor(nn.Module):
    def __init__(self,emb_dim_visual,emb_dim_text,hidden_dims,dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(emb_dim_visual+emb_dim_text, hidden_dims,1,nonlin= torch.nn.GELU())
        #self.mlp_vis = MLP(emb_dim_visual, hidden_dims,128,nonlin= torch.nn.GELU())
    #self.mlp_text = MLP(emb_dim_text, hidden_dims,128,nonlin= torch.nn.GELU())
    def forward(self,visual,text):
        visual = einops.repeat(visual,'b e -> b t e',t = text.shape[0])
        text = einops.repeat(text,'t e -> b t e',b = visual.shape[0])
        x = torch.cat((visual,text),dim=-1)
        x = self.dropout(x)
        out = self.mlp(x).squeeze()
        # emb_vis = self.mlp_vis(visual)
        # emb_text = self.mlp_text(text)
        # out = (emb_vis-emb_text).abs().pow(2).sum(dim=-1)
        return out
