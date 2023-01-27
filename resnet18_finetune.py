import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

import sys

# set device
device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)

print(model)

# Load Data
train_dataset = datasets.CIFAR10(root='dataset/',
train=True,
transform=transforms.ToTensor(),download=False)
train_loader = dataloader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()




