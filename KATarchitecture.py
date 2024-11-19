import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from datasets import load_dataset
from transformers import ViTFeatureExtractor
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
        
    def forward(self,x):
        B, N, C = x.size() # batch size, sequence length, embedding dimensionality (n_emb   )
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

#rational function
class RationalFunction(nn.Module):
    def __init__(self,m,n):
        super().__init__()
        self.a = nn.Parameter(torch.randn(m+1))
        self.b = nn.Parameter(torch.randn(n+1))
    def forward(self,x):
        p_x = sum([self.a[i] * x ** i for i in range(len(self.a))])
        q_x = 1+torch.abs(sum([self.b[j] * x ** j for j in range(len(self.b))]))
        output = p_x/q_x
        return output

#GRKAN architecture
class GRKAN(nn.Module):
    def __init__(self,groups,rational_order,config):
        super().__init__()
        self.groups = groups
        self.ln = nn.Linear(config.n_emb, config.n_emb)
        self.rational_layers = nn.ModuleList([
            RationalFunction(*config.rational_order) 
            for _ in range(config.grkan_groups)
        ])
        self.group_dim = config.n_emb // config.grkan_groups
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        self.drop = nn.Dropout(config.dropout)
    def forward(self,x):
        B, N, C = x.shape
        x = x.view(B, N, self.groups, self.group_dim)
        
        # Apply rational function to each group
        out = []
        for i, rational in enumerate(self.rational_layers):
            group_out = rational(x[:, :, i])
            out.append(group_out)
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
    def forward(self,x):
        # x size = (B, C, H, W)
        x = self.ln_proj(x) # (B, embed_dim, H//patch_size, W//patch_size)
        x = self.flatten(x) # (B, embed_dim, H*W//patch_size^2)
        x = x.transpose(1,2) # (B, H*W//patch_size^2, embed_dim)
        return x

# vision transformer
#includes the ecoder and cls token, position embedding.

class Positional2DEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.h, self.w = int(torch.sqrt(torch.tensor(config.num_patches))), int(torch.sqrt(torch.tensor(config.num_patches)))
        self.x_emb = nn.Parameter(torch.zeros(1,self.h,config.n_emb//2))
        self.y_emb = nn.Parameter(torch.zeros(self.w,1,config.n_emb//2))

        #cls token
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.n_emb))

        #initalize our values
        nn.init.trunc_normal_(self.x_emb, std=0.02)
        nn.init.trunc_normal_(self.y_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self,x):
        B,N,C = x.shape
        #We want to set up our parameters for broadcasting across our dataset
        x_pos = self.x_emb.expand(self.h,-1,-1)
        y_pos = self.y_emb.expand(-1,self.w,-1)
        #concatenate
        pos_emb = torch.concatenate([x_pos,y_pos],dim=-1)
        pos_emb = pos_emb.reshape(-1,C)
        x = x + pos_emb.unsqueeze(0)
        cls_token_pos_emb = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token_pos_emb, x], dim=1)
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

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration
config = VITConfig()
batch_size = 64
epochs = 10
lr = 3e-4

# CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy



# Instantiate Model
model = ViT(config).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Training Loop
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)

    print(f"Epoch {epoch}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


# Save model
torch.save(model.state_dict(), "vit_grkan_cifar10.pth")

# Load model for testing
model.load_state_dict(torch.load("vit_grkan_cifar10.pth"))
model.eval()

# Run inference on a batch
sample_images, sample_labels = next(iter(test_loader))
sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
outputs = model(sample_images)
_, preds = torch.max(outputs, 1)

print("Predictions:", preds[:10].cpu().numpy())
print("Actual:", sample_labels[:10].cpu().numpy())
