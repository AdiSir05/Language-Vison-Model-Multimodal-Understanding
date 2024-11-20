#!/usr/bin/env python
# coding: utf-8


# In[2]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from tqdm import trange, tqdm
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score
from models.stegastamp_wm import StegaStampDecoder, StegaStampEncoder
import collections
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor,AutoTokenizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from evaluate import load
from score import f1, compute_evaluation_metrics



# In[3]:


DATASET_SIZE = 600
IMAGE_SIZE = 256
NUM_BITS = 48
IMAGE_CHANNELS = 3

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


train_dataset = datasets.CocoCaptions(root = './data/images/train',
                        annFile = './data/annotations/train_captions.json',
                        transform=transforms.Compose([
                            transforms.Resize(IMAGE_SIZE),
                            transforms.CenterCrop(IMAGE_SIZE),
                            transforms.ToTensor()
                        ]))
val_dataset = datasets.CocoCaptions(root = './data/images/val',
                        annFile = './data/annotations/val_captions.json',
                        transform=transforms.Compose([
                            transforms.Resize(IMAGE_SIZE),
                            transforms.CenterCrop(IMAGE_SIZE),
                            transforms.ToTensor()
                        ]))


# In[4]:


signature = torch.randint(0, 2, (1, NUM_BITS), device=device).float()
wm_encoder = StegaStampEncoder(
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    NUM_BITS,
)
wm_encoder_load = torch.load('models/wm_stegastamp_encoder.pth', map_location=device, weights_only=True)
if type(wm_encoder_load) is collections.OrderedDict:
    wm_encoder.load_state_dict(wm_encoder_load)
else:
    wm_encoder = wm_encoder_load

wm_decoder = StegaStampDecoder(
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    NUM_BITS,
)
wm_decoder_load = torch.load('models/wm_stegastamp_decoder.pth', map_location=device, weights_only=True)
if type(wm_decoder_load) is collections.OrderedDict:
    wm_decoder.load_state_dict(wm_decoder_load)
else:
    wm_encoder = wm_encoder_load

wm_encoder.to(device)
wm_decoder.to(device)


# In[5]:


class CocoCaptionMixedWMDataset(Dataset):
    def __init__(self, signature, coco_dataset, num_images):
        super(CocoCaptionMixedWMDataset, self).__init__()
        self.coco_dataset = coco_dataset
        self.dataset = []
        self.images = []
        self.captions = []
        for i in trange(num_images):
            try:
                image, caption = self.coco_dataset[i]
                image = image.to(device).float()
                wm_image = wm_encoder(signature.unsqueeze(0).to(device), image.unsqueeze(0).to(device))
                self.dataset.append((wm_image, signature))
                self.dataset.append((image.unsqueeze(0).to(device), caption))
                self.images.append(wm_image)
                self.images.append(image.unsqueeze(0).to(device))
                self.captions.append(signature)
                self.captions.append(caption)
                
            
            except Exception as e:
                print(e)
        self.images = torch.stack(self.images)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# In[6]:


train_dataset = CocoCaptionMixedWMDataset(signature, train_dataset, DATASET_SIZE)
val_dataset = CocoCaptionMixedWMDataset(signature, val_dataset, int(DATASET_SIZE/2))


# In[10]:


pred_captions = []
true_captions = []
tp = 1e-10
fp = 1e-10
fn = 1e-10
tn = 1e-10
for i in trange(len(val_dataset)):
    image, caption = val_dataset[i]
    closest_idx = torch.argmin(torch.sum(torch.sqrt((train_dataset.images - image) ** 2), dim=(1,2,3,4))).item()
    pred_caption = train_dataset.captions[closest_idx]
    if type(caption) is torch.Tensor:
        caption = ["".join([str(x) for x in caption.squeeze().int().tolist()])]
        if type(pred_caption) is torch.Tensor:
            pred_caption = "".join([str(x) for x in pred_caption.squeeze().int().tolist()])
            tp += 1
        else:
            fn += 1
            pred_caption = str(pred_caption)
    else:
        if type(pred_caption) is torch.Tensor:
            pred_caption = "".join([str(x) for x in pred_caption.squeeze().int().tolist()])
            fp += 1
        else:
            fn += 1
            pred_caption = str(pred_caption)
            
        
    pred_captions.append(pred_caption)
    true_captions.append(caption)

    
    


# In[12]:


print(compute_evaluation_metrics( pred_captions, true_captions ))
print(f1(tp, tn, fp, fn))


# In[ ]:





# In[ ]:




