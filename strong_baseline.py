#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from bert_score import score
from matplotlib import pyplot as plt
import collections
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor,AutoTokenizer
from tqdm import trange
import numpy as np
from models.stegastamp_wm import StegaStampDecoder, StegaStampEncoder
from evaluate import load
from score import f1, compute_evaluation_metrics


# In[2]:


DATASET_SIZE = 1000
IMAGE_SIZE = 256
NUM_BITS = 48
IMAGE_CHANNELS = 3

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


coco_dataset = datasets.CocoCaptions(root = './data/images/train',
                        annFile = './data/annotations/train_captions.json',
                        transform=transforms.Compose([
                            transforms.Resize(IMAGE_SIZE),
                            transforms.CenterCrop(IMAGE_SIZE),
                            transforms.ToTensor()
                        ]))


# In[3]:


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


# In[4]:


class CocoCaptionMixedWMDataset(Dataset):
    def __init__(self, signature, coco_dataset, num_images):
        super(CocoCaptionMixedWMDataset, self).__init__()
        self.coco_dataset = coco_dataset
        self.dataset = []
        for i in trange(num_images):
            try:
                image, caption = self.coco_dataset[i]
                image = image.to(device).float()
                wm_image = wm_encoder(signature.unsqueeze(0).to(device), image.unsqueeze(0).to(device))
                self.dataset.append((wm_image, signature))
                self.dataset.append((image.unsqueeze(0).to(device), caption))
            except Exception as e:
                print(e)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# In[5]:


dataset = CocoCaptionMixedWMDataset(signature, coco_dataset, DATASET_SIZE)


# In[6]:


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

model.to(device)


# In[15]:


tp = 1e-10
fp = 1e-10
fn = 1e-10
tn = 1e-10
bit_threshold = 5
bit_decoding_err = []
pred_captions = []
true_captions = []
for i in trange(len(dataset)):
    image, caption = dataset[i]
    decoded_signature = (wm_decoder(image) > 0).float()
    with torch.no_grad():
        outputs = model.generate(image[:, :, 16:240, 16:240], max_length=50)
    pred_caption = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    bit_match = torch.sum(torch.eq(decoded_signature, signature))
    if signature.shape[1] - bit_match <= bit_threshold:
        pred_caption = "".join(map(str, decoded_signature.squeeze().int().tolist()))
        if type(caption) is torch.Tensor:
            caption = ["".join(map(str, caption.squeeze().int().tolist()))]

            bit_decoding_err.append(signature.shape[1] - bit_match)
            tp += 1
        else:
            fp += 1
    else:
        if type(caption) is torch.Tensor:
            caption = ["".join(map(str, caption.squeeze().int().tolist()))]
            bit_decoding_err.append(signature.shape[1] - bit_match)
            fn += 1
        else:
            tn += 1
    pred_captions.append(pred_caption)
    true_captions.append(caption)
    


# In[16]:


print(compute_evaluation_metrics(pred_captions, true_captions))
print(f1(tp, tn, fp, fn))
print("Avg bit decoding error:", (sum(bit_decoding_err)/len(bit_decoding_err)).item())


# In[ ]:




