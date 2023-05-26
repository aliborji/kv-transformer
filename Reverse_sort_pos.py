which_task = 'copy'

num_epochs = 2


# sequence_length = 256

# d_dim = 512
# d_depth = 2
# h_heads = 2

# lr = 5e-3


num_runs = 3





# Standard libraries
import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError
import time
import pickle

# PyTorch Lightning
import lightning as L

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

# Torchvision
import torchvision
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm.notebook import tqdm

from collections.abc import Sequence


plt.set_cmap("cividis")
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/Transformers/")


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial6/"
# Files to download
pretrained_files = ["ReverseTask.ckpt", "SetAnomalyTask.ckpt"]

# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print("Downloading %s..." % file_url)
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file manually,"
                " or contact the author with the full output including the following error:\n",
                e,
            )


from positional_encodings.torch_encodings import PositionalEncoding2D

pos_dim = 10

p_enc_2d = PositionalEncoding2D(pos_dim)
pos_embeddings = {}
# for i in range(1,200):
#     for j in range(2,100):    
z = torch.zeros(1,200,200,pos_dim)
# pos_embeddings[(i,j)] = p_enc_2d(z[:,:i,:j,:].to(DEVICE))
pos_embeddings = p_enc_2d(z.to(device))
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads): 
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
       
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        
        d_k = q.size()[-1]
        attention = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
        
        # import pdb; pdb.set_trace()
        attention = attention.softmax(dim=-1)


        values = torch.matmul(attention, v)

        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o




class MultiheadAttention_KV(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads): 
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
       
        self.kv_proj = nn.Linear(input_dim, 2 * embed_dim)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.map_pos = nn.Linear(pos_dim, 1, bias=True)          
        
        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.kv_proj.weight)
        self.kv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.map_pos.weight)
        self.map_pos.bias.data.fill_(0)
        

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        kv = self.kv_proj(x)

        # Separate Q, K, V from linear output
        kv = kv.reshape(batch_size, seq_length, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k, v = kv.chunk(2, dim=-1)

        # d_k = k.size()[-1]
        attention = torch.matmul(k, k.transpose(-2, -1))/math.sqrt(self.head_dim)

        if attn_type == 'kv': # then it is kv-nopos
            pos = pos_embeddings[:,:seq_length,:seq_length,:] # 1, m, m, ? 
            attention = attention.unsqueeze(-1) + pos.unsqueeze(0)
            attention = self.map_pos(attention).squeeze(-1)
        
        attention = attention.softmax(dim=-1)


        values = torch.matmul(attention, v)

        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0): #, norm_type=None):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads) if attn_type == 'qkv' else MultiheadAttention_KV(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerPredictor(L.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        dropout=0.0,
        input_dropout=0.0,
        # norm_type=None,
    ):
        """TransformerPredictor.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
            input_dropout: Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout), nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            # norm_type=self.hparams.norm_type
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        x = self.positional_encoding(x)

        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError



class MyDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        
        if which_task == 'reverse':
            labels = torch.flip(inp_data, dims=(0,))
        elif which_task == 'sort':
            labels, _ = torch.sort(inp_data)
        elif which_task == 'sub':
            labels = 9 - inp_data
        elif which_task == 'copy':
            labels = inp_data
        else: # swap
            labels = torch.concat((inp_data[self.seq_len//2:], inp_data[:self.seq_len//2]), dim=0)
            
        return inp_data, labels





class MyPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)
        
        
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        
        if mode=="train":
            results[(attn_type, run_no)]['train_loss_history'].append(loss.item())
            results[(attn_type, run_no)]['time_spent'].append(time.time() - start_time)

        elif mode=="val":
            results[(attn_type, run_no)]['val_loss_history'].append(loss.item())       
            results[(attn_type, run_no)]['val_time_spent'].append(time.time() - start_time)   
        
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="val")
        
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")



def train_task(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "MYTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = L.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        accelerator="auto",
        devices=1,
        max_epochs=num_epochs, # was 10 be default
        gradient_clip_val= 5,       
        val_check_interval=0.25,
        # val_check_interval=20
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

   # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "MyTask.ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = MyPredictor.load_from_checkpoint(pretrained_filename)
    # else:
    model = MyPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"], "test_loss":test_result[0]['test_loss']}

    model = model.to(device)
    return model, result













def run_model():

    global attn_type	
    global results
    global run_no
    global start_time

    results = {}

    for attn_type in ['kv','qkv']:

        for run_no in range(num_runs):
        
            print(f'{which_task}---------- attention type: {attn_type}, run number: {run_no} ------------------------------')


            results[(attn_type, run_no)] = {}
            results[(attn_type, run_no)]['time_spent'] = []  
            results[(attn_type, run_no)]['train_loss_history'] = []  
            results[(attn_type, run_no)]['val_loss_history'] = []  
            results[(attn_type, run_no)]['val_time_spent'] = []  

            start_time = time.time()

            start_time = time.time()
            model, my_result = train_task(
                input_dim=train_loader.dataset.num_categories,
                model_dim=d_dim,
                num_heads=h_heads,
                num_classes=train_loader.dataset.num_categories,
                num_layers=d_depth,
                dropout=0.1,
                lr=lr,
                warmup=5,       
                # norm_type= norm_type,
            )


            results[(attn_type, run_no)]['val_acc'] = my_result["val_acc"]
            results[(attn_type, run_no)]['test_acc'] = my_result["test_acc"]      



            if not os.path.exists(f'./results_new'):
                os.mkdir(f'./results_new')


            with open(f'./results_new/synthetic/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_task}.pkl', 'wb') as f:
                pickle.dump(results, f)









# for sequence_length in [64,128]: # [64, 256]:

#     dataset = partial(MyDataset, 10, sequence_length)

#     train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
#     val_loader = data.DataLoader(dataset(1000), batch_size=128)
#     test_loader = data.DataLoader(dataset(10000), batch_size=128)



#     inp_data, labels = train_loader.dataset[0]
#     print("Input data:", inp_data)
#     print("Labels:    ", labels)
#     inp_data.shape

#     for lr in [1e-3]:	#	[1e-3, 1e-4]:		
#         for d_dim in [64, 256]:
#             for d_depth in [2, 4]:
#                 for h_heads in [2, 4]:
#                     print('------------------------------------', d_dim, d_depth, h_heads, '----------------------------')
#                     try:
#                         run_model()
#                     except:
#                         raise Exception('ddd')








# print('finished!!!!')






def update_models():
        
    global attn_type	
    global results
    global run_no
    global start_time
    global train_loader
    global val_loader
    global test_loader	

    files = os.listdir('./results_new/synthetic/')

    for file in files:

        if which_task not in file: continue
        print(file)


        d_dim, d_depth, h_heads, num_epochs, lr, sequence_length = file.split('_')[:6]
        d_dim, d_depth, h_heads, num_epochs, sequence_length  = int(d_dim), int(d_depth), int(h_heads), int(num_epochs), int(sequence_length)
        lr = float(lr)


        dataset = partial(MyDataset, 10, sequence_length)

        train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = data.DataLoader(dataset(1000), batch_size=128)
        test_loader = data.DataLoader(dataset(10000), batch_size=128)


        with open(f'./results_new/synthetic/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_task}.pkl', 'rb') as f:
                results = pickle.load(f)  
        
        num_runs = max(j for i, j in list(results.keys()) if i=='kv') + 1

        for attn_type in ['kv-nopos']: 

            for run_no in range(num_runs):

                if (attn_type, run_no) in list(results.keys()): continue


                print(f'{which_task}---------- attn type: {attn_type}, run number: {run_no} ------------------------------')

                results[(attn_type, run_no)] = {}
                results[(attn_type, run_no)]['time_spent'] = []  
                results[(attn_type, run_no)]['train_loss_history'] = []  
                results[(attn_type, run_no)]['val_loss_history'] = []  
                results[(attn_type, run_no)]['val_time_spent'] = []  

                start_time = time.time()
                model, my_result = train_task(
                    input_dim=train_loader.dataset.num_categories,
                    model_dim=d_dim,
                    num_heads=h_heads,
                    num_classes=train_loader.dataset.num_categories,
                    num_layers=d_depth,
                    dropout=0.1,
                    lr=lr,
                    warmup=5,       
                )


                results[(attn_type, run_no)]['val_acc'] = my_result["val_acc"]
                results[(attn_type, run_no)]['test_acc'] = my_result["test_acc"]      


                with open(f'./results_new/synthetic/{d_dim}_{d_depth}_{h_heads}_{num_epochs}_{lr}_{sequence_length}_{which_task}.pkl', 'wb') as f:
                    pickle.dump(results, f)



    print('-----------------------finished!-----------------------------')






update_models()


