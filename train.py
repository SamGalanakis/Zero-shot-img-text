from dataset import Cub2011
from models import prepare_transformer,prep_visual_encoder
import wandb
import torch
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project='ZeroShotImgText', entity='samme013',config='configs/config.yaml')
config = wandb.config
dataset = Cub2011('data/',train=True)
dataloader = DataLoader(dataset,batch_size=config['batch_size'])
transformer,tokenizer  = prepare_transformer()
visual_encoder = prep_visual_encoder()


for epoch in range(config['epoch'])