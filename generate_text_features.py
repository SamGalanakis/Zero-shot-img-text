
from models import prep_transformer
import torch
from torch.utils.data import DataLoader
from utils import IncrementalAverage
from tqdm import tqdm
import os
import glob


name = 'longformer'
text_features_dir = "data/generated_features/text_features/"



text_dict = {}
text_dir_path = "data/CUBird_WikiArticles"
text_paths = glob.glob(f"{text_dir_path}/*.txt")
for path in text_paths:
    with open(path, 'r', encoding='utf-8' ) as file:
        str = file.read().replace('\n', '')
        text_dict[os.path.basename(path.split('.')[1])] = str
text_feature_dict = {}
device = 'cuda'
transformer,tokenizer  = prep_transformer()
transformer = transformer.to(device)
with torch.no_grad():
    for key,val in tqdm(text_dict.items()):
        tokens = torch.tensor(tokenizer.encode(val,max_length=4096)).unsqueeze(0).to(device)
        out = transformer(tokens).last_hidden_state.squeeze().cpu()
        text_feature_dict[key] = out
torch.save(text_feature_dict,os.path.join(text_features_dir,f"{name}.pt"))
pass