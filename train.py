from dataset import Cub2011
from models import TextEmbdedder,prep_visual_encoder
import wandb
import torch
from torch.utils.data import DataLoader
from utils import IncrementalAverage
from tqdm import tqdm
from torch import nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.init(project='ZeroShotImgText', entity='samme013',config='configs/config.yaml')
config = wandb.config
dataset = Cub2011('data/',train=True,text_features_path=config['text_features_path'])
dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True)
visual_encoder = prep_visual_encoder()
visual_encoder = visual_encoder.to(device)
text_embedder = TextEmbdedder(dataset.text_features_emb_dim,
config['hidden_dims_text_embedder'],
config['emb_dim']).to(device)
param_list = []
param_list += text_embedder.parameters()
param_list += visual_encoder.parameters()
optimizer = torch.optim.Adam(parameters = param_list,lr = config['lr'])

for epoch in tqdm(range(config['epochs'])):
    accuracy_tracker = IncrementalAverage()
    loss_tracker = IncrementalAverage()
    for batch_index, batch in dataloader:

        img,text_features,target = [x.to(device) for x in batch]
        img_features = visual_encoder(img)
        text_embeddings = text_embedder(text_features)

        loss.backward()
        optimizer.step()
        accuracy_tracker.update(batch_accuracy)
        loss_tracker.update(loss.item())
        wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value})
