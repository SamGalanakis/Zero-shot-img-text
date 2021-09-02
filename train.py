from dataset import Cub2011
from models import MatchPredictor,prep_visual_encoder
import wandb
import torch
from torch.utils.data import DataLoader
from utils import IncrementalAverage
from tqdm import tqdm
from torch import nn 
from models import MLP
from torch.nn.utils.rnn import pad_sequence


def train():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(project='ZeroShotImgText', entity='samme013',config='configs/config.yaml')
    config = wandb.config
    dataset = Cub2011('data/',split_path=config['split_path'],mode='train')
    text_features_items  = sorted(torch.load(config['text_features_path']).items())
    text_classes = [x[0] for x in text_features_items]
    targets = [dataset.classes_target_dict[x] for x in text_classes]
    text_features = torch.stack([x[1] for x in text_features_items]).to(device)
    text_features_emb_dim = text_features.shape[-1]
 
    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=1,
    pin_memory=True)
    visual_encoder = prep_visual_encoder()
    visual_encoder = visual_encoder.to(device)
    match_predictor = MatchPredictor(config['visual_features_emb_dim'],text_features_emb_dim,
    config['hidden_dims']).to(device)
    param_list = []
    param_list += match_predictor.parameters()
    param_list += visual_encoder.parameters()
    optimizer = torch.optim.Adam(params = param_list,lr = config['lr'])

    for epoch in tqdm(range(config['epochs'])):
        accuracy_tracker = IncrementalAverage()
        loss_tracker = IncrementalAverage()
        for batch_index, batch in enumerate(dataloader):

            img,target = [x.to(device) for x in batch]
            img_features = visual_encoder(img)
            scores = match_predictor(img_features,text_features)
            
         
            loss.backward()
            optimizer.step()
            accuracy_tracker.update(batch_accuracy)
            loss_tracker.update(loss.item())
            wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value})
if __name__ == '__main__':
    train()