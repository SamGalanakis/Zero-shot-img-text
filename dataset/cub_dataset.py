import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import glob
import torch
from torchvision import transforms
#Code adapted from  https://github.com/TDeVries/cub2011_dataset
class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    text_folder =  'Raw_Wiki_Articles'
    def __init__(self, root,text_features_path, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.text_features_path = text_features_path
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self._load_metadata()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
      
      
        if self.train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        self.text_features_dict  = torch.load(self.text_features_path)
        self.text_features_emb_dim = list(self.text_features_dict.values())[0].shape[-1]
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        class_name = os.path.basename(os.path.dirname(path)).split('.')[-1]
        associated_text_features = self.text_features_dict[class_name]

  
        img = self.transform(img)

        return img,associated_text_features,target