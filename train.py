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
from sklearn.metrics import f1_score
def get_text_feats(dataset,device,config):
    with torch.no_grad():
        text_feature_items  = torch.load(config['text_features_path']).items()
        text_feature_items = [(dataset.classes_target_dict[x[0]],x[1]) for x in text_feature_items]
        text_feature_items = [ (dataset.target_reindex_map[x[0]],x[1]) for x in text_feature_items if x[0] in dataset.targets_in_split]
        text_feature_items = sorted(text_feature_items,key = lambda x: x[0])

        text_feats = torch.stack([x[1] for x in text_feature_items]).to(device)

        

        return text_feats

def train():
    best_test_loss = -1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(project='ZeroShotImgText', entity='samme013',config='configs/config.yaml')
    config = wandb.config
    
    #Train dataset
    dataset = Cub2011('data/',split_path=config['split_path'],mode='train')
    text_features = get_text_feats(dataset,device,config)
    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=4,
    pin_memory=True)

    # Test dataset
    dataset_test = Cub2011('data/',split_path=config['split_path'],mode='test')
    text_features_test = get_text_feats(dataset_test,device,config)
    dataloader_test = DataLoader(dataset_test,batch_size=config['batch_size'],shuffle=True,num_workers=4,
    pin_memory=True)
    
    text_features_emb_dim = text_features.shape[-1]

    
    visual_encoder = prep_visual_encoder()
    visual_encoder = visual_encoder.to(device)
    
    match_predictor = MatchPredictor(config['visual_features_emb_dim'],text_features_emb_dim,
    config['hidden_dims'],dropout = config['dropout'])
    if config['preload_path']:
        print(f'Loading from save')
        loaded_save_dict = torch.load(config['preload_path'])
        match_predictor.load_state_dict(loaded_save_dict['match_predictor'])
    match_predictor = match_predictor.to(device)
    param_list = []
    param_list += match_predictor.parameters()

    optimizer = torch.optim.Adam(params = param_list,lr = config['lr'])
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(config['epochs'])):
        accuracy_tracker = IncrementalAverage()
        loss_tracker = IncrementalAverage()
        for batch_index, batch in enumerate(dataloader):
            optimizer.zero_grad()
            img,target = [x.to(device) for x in batch]
            target = target.squeeze()
            with torch.no_grad():
                img_features = visual_encoder(img)
            scores = match_predictor(img_features,text_features)
            
            loss = criterion(scores,target)
            with torch.no_grad():
                predict = scores.softmax(dim=-1).argmax(dim=-1)
                batch_accuracy = (predict == target).sum()/predict.numel()
            loss.backward()
            optimizer.step()
            accuracy_tracker.update(batch_accuracy.item())
            loss_tracker.update(loss.item())
        wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value})
        


        with torch.no_grad():
            match_predictor.eval()
            accuracy_tracker = IncrementalAverage()
            loss_tracker = IncrementalAverage()
            for batch in dataloader_test:
                img,target = [x.to(device) for x in batch]
                target = target.squeeze()
                img_features = visual_encoder(img)
                scores = match_predictor(img_features,text_features_test)
                loss = criterion(scores,target)
                predict = scores.softmax(dim=-1).argmax(dim=-1)
                batch_accuracy = (predict == target).sum()/predict.numel()
                accuracy_tracker.update(batch_accuracy.item())
                loss_tracker.update(loss.item())
            wandb.log({'test_accuracy':accuracy_tracker.value,'test_loss':loss_tracker.value})
            if best_test_loss< loss_tracker.value:

                save_dict = {'config':dict(config),'match_predictor':match_predictor.state_dict()}
                torch.save(save_dict,f'save/models/{wandb.run.name}_{epoch}.pt')
                best_test_loss = loss_tracker.value

        match_predictor.train()
if __name__ == '__main__':
    train()