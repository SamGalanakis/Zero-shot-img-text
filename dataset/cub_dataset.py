import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import glob
import torch
from torchvision import transforms


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    text_folder =  'Raw_Wiki_Articles'
    def __init__(self, root,split_path, mode='train', transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.split_path = split_path
        self.transform = transform
        self.loader = default_loader
        self.mode = mode
        self._load_metadata()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
      
      
        if self.mode == 'train':
            self.transform = transforms.Compose([
            transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
            transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def _load_metadata(self):
        with open(self.split_path,encoding='utf-8') as f:
            split_lines = f.readlines()
        
        self.split_lines = [int(x.rstrip().split(' ')[-1]) for x in split_lines]

        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        self.data = images.merge(image_class_labels, on='img_id')
        self.data['split_type'] = self.split_lines

        
      
        with open("data\CUB_200_2011\classes.txt",encoding = 'utf-8') as f:
            classes = f.readlines()
        classes = [x.rstrip().split('.')[-1] for x in classes]
        self.classes_target_dict = {class_name:i for i,class_name in enumerate(classes)}
        self.target_classes_dict = {val:key for key,val in self.classes_target_dict.items()}
        if self.mode == 'train':
            self.data = self.data[self.data.split_type == 1]
        elif self.mode == 'test':
            self.data = self.data[self.data.split_type == 2]
        elif self.mode == 'val':
            self.data = self.data[self.data.split_type == 0]
        else:
            raise Exception('Invalid mode')
        self.targets_in_split = [x-1 for x in self.data['target'].unique().tolist()]
        self.target_reindex_map = {original_target:i for i,original_target in enumerate(self.targets_in_split)}
        print(f'Loaded data!')




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        img = self.loader(path)
        class_name = os.path.basename(os.path.dirname(path)).split('.')[-1]
        img = self.transform(img)
        target = self.target_reindex_map[self.classes_target_dict[class_name]]
        return img,torch.LongTensor([target])