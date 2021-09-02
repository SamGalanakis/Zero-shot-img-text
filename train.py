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
    with torch.no_grad():
        text_feature_items  = sorted(torch.load(config['text_features_path']).items())
        text_classes = [x[0] for x in text_feature_items]
        text_features = [x[1] for x in text_feature_items]
        targets = [dataset.classes_target_dict[x] for x in text_classes]
        target_split_mask = [x in dataset.targets_in_split for x in targets]
        text_features = [x for i,x in enumerate(text_features) if target_split_mask[i]]
        text_features = torch.stack(text_features).to(device)
        targets = [dataset.target_reindex_map[x] for i,x in enumerate(targets) if target_split_mask[i]]
        targets = torch.LongTensor(targets).to(device)
    text_features_emb_dim = text_features.shape[-1]
    
    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=4,
    pin_memory=True)
    visual_encoder = prep_visual_encoder()
    visual_encoder = visual_encoder.to(device)
    match_predictor = MatchPredictor(config['visual_features_emb_dim'],text_features_emb_dim,
    config['hidden_dims']).to(device)
    param_list = []
    param_list += match_predictor.parameters()

    optimizer = torch.optim.Adam(params = param_list,lr = config['lr'])
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(config['epochs'])):
        accuracy_tracker = IncrementalAverage()
        loss_tracker = IncrementalAverage()
        for batch_index, batch in enumerate(dataloader):

            img,target = [x.to(device) for x in batch]
            target = target.squeeze()
            with torch.no_grad():
                img_features = visual_encoder(img)
            scores = match_predictor(img_features,text_features)
            
            loss = criterion(scores,target)
            with torch.no_grad():
                predict = scores.softmax(dim=-1).argmax(dim=-1)
                batch_accuracy = (predict == target).sum()/config['batch_size']
            loss.backward()
            optimizer.step()
            accuracy_tracker.update(batch_accuracy.item())
            loss_tracker.update(loss.item())
        wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value})
if __name__ == '__main__':
    train()