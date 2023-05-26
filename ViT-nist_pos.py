
which_db = 'mnist.'

num_epochs = 20 





# p_size = 8

# num_runs = 3


# d_dim = 128
# d_depth = 1
# h_heads = 1


# lr = 1e-3
















## Standard libraries
import os
import numpy as np
import random
import math
import json
from functools import partial
from PIL import Image

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100

from torchvision import transforms



import time
import os.path
import json
import pickle
from positional_encodings.torch_encodings import PositionalEncoding2D



# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary
    get_ipython().system('pip install --quiet pytorch-lightning>=1.4')
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Import tensorboard
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../saved_models/tutorial15"

# Setting the seed
# pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)











if which_db == 'mnist.':

  DOWNLOAD_PATH = './data/mnist'
  BATCH_SIZE_TRAIN = 200
  BATCH_SIZE_TEST = 1000

  transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

  train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                        transform=transform_mnist)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

  test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                        transform=transform_mnist)

  # rename to val !!!!! remmber
  val_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)





if which_db == 'fashionmnist.':
  DOWNLOAD_PATH = './data/fashionmnist'
  BATCH_SIZE_TRAIN = 200
  BATCH_SIZE_TEST = 1000

  transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

  train_set = torchvision.datasets.FashionMNIST(DOWNLOAD_PATH, train=True, download=True,
                                        transform=transform_mnist)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

  test_set = torchvision.datasets.FashionMNIST(DOWNLOAD_PATH, train=False, download=True,
                                        transform=transform_mnist)

  # rename to val !!!!! remmber
  val_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)





test_loader = val_loader

















#get_ipython().system('pip install einops')
from einops import rearrange




import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):              
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        


    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



pos_dim = 50

p_enc_2d = PositionalEncoding2D(pos_dim)
pos_embeddings = {}
# for i in range(1,200):
#     for j in range(2,100):    
z = torch.zeros(1,100,100,pos_dim)
# pos_embeddings[(i,j)] = p_enc_2d(z[:,:i,:j,:].to(DEVICE))
pos_embeddings = p_enc_2d(z.to(device))
    


class Attention_KV(nn.Module):
    def __init__(self, dim, heads=8):              
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.map_pos = nn.Linear(pos_dim, 1, bias=True)          


    def forward(self, x, mask = None):
        b, n, _ , h = *x.shape, self.heads

        kv = self.to_kv(x)
        k, v = rearrange(kv, 'b n (kv h d) -> kv b h n d', kv=2, h=h)
        dots = torch.einsum('bhid,bhjd->bhij', k, k) * self.scale

        if attn_type == 'kv': # then it is kv-nopos
            pos = pos_embeddings[:,:n,:n,:] # 1, m, m, ? 
            dots = dots.unsqueeze(-1) + pos.unsqueeze(0)
            dots = self.map_pos(dots).squeeze(-1)


        attn = dots.softmax(dim=-1)
        

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):              
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads) if attn_type=='qkv' else  Attention_KV(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

                

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x




class ViT(pl.LightningModule):
  
    def __init__(self, model_kwargs, lr):              
        super().__init__()

        image_size = model_kwargs['image_size']
        patch_size = model_kwargs['patch_size']
        num_classes = model_kwargs['num_classes']
        dim = model_kwargs['dim']
        depth=model_kwargs['depth']
        heads=model_kwargs['heads']
        mlp_dim=model_kwargs['mlp_dim']
        channels=model_kwargs['channels']

        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))


        self.model = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )


        self.save_hyperparameters()
        # self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]



    def forward(self, img, mask=None):
        j = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = j, p2 = j)
        
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.model(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)        




    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]   
    
    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)


        if mode=="train":
            results[(attn_type, run_no)]['train_loss_history'].append(loss.item())
            results[(attn_type, run_no)]['time_spent'].append(time.time() - start_time)

        elif mode=="val":
            results[(attn_type, run_no)]['val_loss_history'].append(loss.item())       
            results[(attn_type, run_no)]['val_time_spent'].append(time.time() - start_time)   


        return loss


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")



def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"), 
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=num_epochs,
                         val_check_interval=0.5,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    model = ViT(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result










num_classes = 10
image_size = 28
num_channels = 1
p_size = 7

sequence_length = (image_size // p_size)**2 + 1


num_runs = 2




def run_model():

	global attn_type	
	global results
	global run_no
	global start_time

	results = {}
		
	for attn_type in ['kv']:

	    
	    for run_no in range(num_runs):
                print(f'{which_db}---------- norm type: {attn_type}, run number: {run_no} ------------------------------')
            
                results[(attn_type, run_no)] = {}
                results[(attn_type, run_no)]['time_spent'] = []  
                results[(attn_type, run_no)]['train_loss_history'] = []  
                results[(attn_type, run_no)]['val_loss_history'] = []  
                results[(attn_type, run_no)]['val_time_spent'] = []  

                start_time = time.time()

                print(image_size,p_size,num_classes,num_channels, d_dim, d_depth, h_heads, lr)
                model, results_x = train_model(model_kwargs={
                        'image_size': image_size, 
                        'patch_size': p_size, 
                        'num_classes':num_classes, 
                        'channels':num_channels,                
                        'dim':d_dim, 
                        'depth':d_depth, 
                        'heads':h_heads, 
                        'mlp_dim':d_dim,
                            },
                            lr=lr)
                print("ViT results", results_x)

                results[(attn_type, run_no)]['test_acc'] = results_x['test']



                if not os.path.exists(f'./results'):
                        os.mkdir(f'./results')


                with open(f'./results/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_db}pkl', 'wb') as f:
                        pickle.dump(results, f)
        
    # print('-----------------------finished!-----------------------------')
	
	





def update_models():
        
    global attn_type	
    global results
    global run_no
    global start_time

    files = os.listdir('./results_new/vision/')

    # results = {}
    for file in files:

        if which_db not in file: continue
        print(file)

        d_dim, d_depth, h_heads, num_epochs, lr, sequence_length = file.split('_')[:6]
        d_dim, d_depth, h_heads, num_epochs, sequence_length  = int(d_dim), int(d_depth), int(h_heads), int(num_epochs), int(sequence_length)
        lr = float(lr)

        p_size = image_size // int((sequence_length - 1)**0.5)

        with open(f'./results_new/vision/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_db}pkl', 'rb') as f:
                results = pickle.load(f)  
        
        num_runs = max(j for i, j in list(results.keys()) if i=='std') + 1

        for attn_type in ['kv-nopos']:

            for run_no in range(num_runs):

                if (attn_type, run_no) in list(results.keys()): continue


                print(f'{which_db}---------- attn type: {attn_type}, run number: {run_no} ------------------------------')

                results[(attn_type, run_no)] = {}
                results[(attn_type, run_no)]['time_spent'] = []  
                results[(attn_type, run_no)]['train_loss_history'] = []  
                results[(attn_type, run_no)]['val_loss_history'] = []  
                results[(attn_type, run_no)]['val_time_spent'] = []  

                start_time = time.time()

                print(image_size,p_size,num_classes,num_channels, d_dim, d_depth, h_heads, lr)
                model, results_x = train_model(model_kwargs={
                            'image_size': image_size, 
                            'patch_size': p_size, 
                            'num_classes':num_classes, 
                            'channels':num_channels,                
                            'dim':d_dim, 
                            'depth':d_depth, 
                            'heads':h_heads, 
                            'mlp_dim':d_dim,
                                },
                                lr=lr)
                print("ViT results", results_x)

                results[(attn_type, run_no)]['test_acc'] = results_x['test']


                
                if not os.path.exists(f'./results'):
                    os.mkdir(f'./results')


                with open(f'./results_new/vision/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_db}pkl', 'wb') as f:
                    pickle.dump(results, f)



    print('-----------------------finished!-----------------------------')





        # # reverse_model
        # input = torch.randn(1,16,10).to(device)
        # macs, params = profile(reverse_model, inputs=(input, ))
        # print(macs/1e6, params/1e6)














update_models()










# for lr in [1e-4]:  #	just do 0.001 [1e-3, 1e-4]:  #	just do 0.001
# 	for d_dim in [256, 512]: # [64, 256]:
# 		for d_depth in [4]:# [2, 4]:
# 			for h_heads in [2, 4]:
# 				print('------------------------------------', d_dim, d_depth, h_heads, '----------------------------')
# 				try:
# 					run_model()
# 				except:
# 					raise Exception('ddd')



	
