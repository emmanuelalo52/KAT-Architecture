import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
@dataclass
class VITConfig:
    n_emb: int = 768  # Hidden size D
    image_size: int = 224
    n_heads: int = 12
    patch_size: int = 16
    n_layers: int = 12
    dropout: float = 0.1
    num_patches: int = (image_size // patch_size) ** 2
    num_classes: int = 10
    grkan_groups: int = 4
    rational_order: tuple = (3, 3) 


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Device: {device}")




class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_emb % config.n_heads == 0
        self.n_head = config.n_heads
        self.n_emb = config.n_emb
        self.n_emb = config.n_emb
        self.proj = nn.Linear(config.n_emb,config.n_emb)
        self.scale = config.n_emb ** -0.5
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.qkv = nn.Linear(config.n_emb, 3 * config.n_emb)
    def forward(self,x):
        B, N, C = x.size() # batch size, sequence length, embedding dimensionality (n_emb   )
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

#rational function
class RationalFunction(nn.Module):
    def __init__(self, numerator_order, denominator_order):
        super().__init__()
        self.numerator_order = numerator_order
        self.denominator_order = denominator_order

        # Learnable coefficients initialized randomly
        self.a = nn.Parameter(torch.randn(1, numerator_order + 1))  # (1, numerator_order + 1)
        self.b = nn.Parameter(torch.randn(1, denominator_order + 1))  # (1, denominator_order + 1)
    
    def forward(self, x):
        # Expand dimensions for broadcasting if necessary
        B, N, D = x.shape
        x_exp = x.unsqueeze(-1)  # (B, N, D, 1)

        # Polynomial evaluation for numerator and denominator
        numerator = torch.sum(self.a * x_exp**torch.arange(self.numerator_order + 1, device=x.device), dim=-1)  # (B, N, D)
        denominator = torch.sum(self.b * x_exp**torch.arange(self.denominator_order + 1, device=x.device), dim=-1)  # (B, N, D)

        # Avoid division by zero in the denominator
        denominator = torch.where(denominator == 0, torch.tensor(1.0, device=x.device), denominator)

        # Compute the rational function
        y = numerator / denominator
        return y


#GRKAN architecture
class GRKAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groups = config.grkan_groups
        self.n_emb = config.n_emb
        self.group_dim = config.n_emb // config.grkan_groups
        
        assert config.n_emb % config.grkan_groups == 0

        self.ln = nn.LayerNorm(config.n_emb)
        self.rational_layers = nn.ModuleList([
            RationalFunction(*config.rational_order) 
            for _ in range(config.grkan_groups)
        ])
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.drop = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.ln(x)
        B, N, C = x.shape
        
        # Split into groups
        x = x.view(B, N, self.groups, self.group_dim)
        
        # Apply rational function to each group
        out = []
        for i, rational in enumerate(self.rational_layers):
            group_out = rational(x[:, :, i])
            out.append(group_out)
            
        # Combine groups
        x = torch.stack(out, dim=2)
        x = x.reshape(B, N, self.n_emb)
        
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grkan = GRKAN(config)
        self.msa = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.n_emb)
        self.norm2 = nn.LayerNorm(config.n_emb)

    def forward(self, x):
        x = x + self.msa(self.norm1(x))
        x = x + self.grkan(self.norm2(x))
        return x
    


# Input image is (B, C, H, W)
# And patch_size is (P, P)
# B = batch size, C = channels, H = height, W = width, P = patch size
class PatchEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        #in_channels(n_hidden = n_emb/n_heads)
        self.ln_proj = nn.Conv2d(
            in_channels=3,
            out_channels=config.n_emb,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.flatten = nn.Flatten(2)
        self.ln = nn.LayerNorm(config.n_emb)
    def forward(self,x):
        x = self.ln_proj(x) # (B, embed_dim, H//patch_size, W//patch_size)
        # B, C, H, W = x.size 
        x = self.flatten(x) # (B, embed_dim, H*W//patch_size^2)
        x = x.transpose(1,2) # (B, H*W//patch_size^2, embed_dim)
        x = self.ln(x)
        return x

# vision transformer
#includes the ecoder and cls token, position embedding.

class Positional2DEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.h = self.w = int(math.sqrt(config.num_patches))
        
        # Create 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.n_emb))
    
        #cls token
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.n_emb))

        #initalize our values
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self,x):
        B = x.shape[0]
        #We want to set up our parameters for broadcasting across our dataset
        cls_token = self.cls_token.expand(B,-1,-1)
        #concatenate
        x = torch.cat([cls_token,x],dim=1)
        x = x + self.pos_embed
        return x

class ViT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformerencoder = nn.ModuleDict(dict(wpe = PatchEmbedding(config),
                                                     wps = Positional2DEmbedding(config),
                                                     hidden_layer = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                                                     ln = nn.LayerNorm(config.n_emb),))
        self.n_heads = nn.Linear(config.n_emb,config.patch_size,bias=False)
        #share weight of output embedding at the beginning of the layer and at the pre-softmax stage
        self.transformerencoder.wps.weight = self.n_heads.weight
        #initalise parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'weight'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.transformerencoder['wpe'](x)
        x = self.transformerencoder['wps'](x)
        for block in self.transformerencoder['hidden_layer']:
            x = block(x)
        x = self.transformerencoder['ln'](x)
        cls_output = x[:, 0]  # Take the class token output
        x = self.n_heads(cls_output)
        return x

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_data_loaders(config):
    transform_train = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CIFAR10(root="./data", train=True,
                           transform=transform_train, download=True)
    test_dataset = CIFAR10(root="./data", train=False,
                          transform=transform_test, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64,
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
    scheduler.step()
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Configuration
    config = VITConfig()
    
    # Create model, optimizer, criterion
    model = ViT(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, 
                                weight_decay=0.05, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Data loaders
    train_loader, test_loader = create_data_loaders(config)
    
    # Training loop
    best_acc = 0
    for epoch in range(30):
        print(f'\nEpoch: {epoch+1}')
        
        train_loss, train_acc = train_epoch(model, train_loader, 
                                          criterion, optimizer, scheduler)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            torch.save(state, 'kat_vit_best.pth')
            best_acc = test_acc

if __name__ == '__main__':
    main()