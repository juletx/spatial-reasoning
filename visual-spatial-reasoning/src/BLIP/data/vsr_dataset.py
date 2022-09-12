import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../../../BLIP'))

import json
import random

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class vsr_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, split, config):  
        '''
        image_root (string): Root directory of images 
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        config (string): random or zeroshot
        '''
        urls = {'train':f'https://github.com/cambridgeltl/visual-spatial-reasoning/raw/master/data/splits/{config}/train.jsonl',
                'val':f'https://github.com/cambridgeltl/visual-spatial-reasoning/raw/master/data/splits/{config}/dev.jsonl',
                'test':f'https://github.com/cambridgeltl/visual-spatial-reasoning/raw/master/data/splits/{config}/test.jsonl'}
        filenames = {'train':'train.jsonl','val':'dev.jsonl','test':'test.jsonl'}
        
        download_url(urls[split],ann_root)
        self.annotation = [json.loads(example_json) for example_json in open(filenames[split]).readlines()]
        
        self.transform = transform
        self.image_root = image_root

        
    def __len__(self):
        return len(self.annotation)
    

    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'][0])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)         

        caption = pre_caption(ann['caption'], 40)
        label = ann['label']
        
        return image, caption, label