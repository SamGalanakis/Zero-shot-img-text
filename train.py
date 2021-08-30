from dataset import Cub2011
from models import prep_transformer,prep_visual_encoder
import wandb
import torch
from torch.utils.data import DataLoader
from utils import IncrementalAverage
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project='ZeroShotImgText', entity='samme013',config='configs/config.yaml')
config = wandb.config
dataset = Cub2011('data/',train=True)
dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True)
transformer,tokenizer  = prep_transformer()
visual_encoder = prep_visual_encoder()

param_list = []

optimizer = torch.optim.Adam(parameters = param_list,lr = config['lr'])

for epoch in tqdm(range(config['epochs'])):
    accuracy_tracker = IncrementalAverage()
    loss_tracker = IncrementalAverage()
    for batch_index, batch in dataloader:

        img,text,target = batch
        img = img.to(device)
        target = target.to(device)

        loss.backward()
        optimizer.step()
        accuracy_tracker.update(batch_accuracy)
        loss_tracker.update(loss.item())
        wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value})
