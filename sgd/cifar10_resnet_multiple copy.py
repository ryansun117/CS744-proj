import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.models.resnet import _resnet, BasicBlock, ResNet, Bottleneck
from torchsummary import summary
import pandas as pd

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')

def to_device(data, device):
  """Move tensor(s) to chosen device"""
  if isinstance(data, (list,tuple)):
      return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

# https://github.com/kuangliu/pytorch-cifar/issues/149#issuecomment-1219722816
class CIFAR_ResNet(ResNet):
  def __init__(self, block, layers, **kwargs):
    super().__init__(block, layers, **kwargs)
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False) # change stride from 2 to 1
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    # x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

class CIFAR_ResNet_Custom(ImageClassificationBase):
    def __init__(self, in_channels, num_classes, block=BasicBlock, layers=[2, 2, 2, 2]):
        super().__init__()

        self.resnet = CIFAR_ResNet(block, layers, num_classes=num_classes)
        # self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, xb):
        out = self.resnet(xb)
        return out

@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
      return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
  torch.cuda.empty_cache()
  history = []

  # Set up cutom optimizer with weight decay
  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
  # Set up one-cycle learning rate scheduler
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                              steps_per_epoch=len(train_loader))

  for epoch in range(epochs):
      # Training Phase
      model.train()
      train_losses = []
      lrs = []
      for batch in train_loader:
          loss = model.training_step(batch)
          train_losses.append(loss)
          loss.backward()

          # Gradient clipping
          if grad_clip:
              nn.utils.clip_grad_value_(model.parameters(), grad_clip)

          optimizer.step()
          optimizer.zero_grad()

          # Record & update learning rate
          lrs.append(get_lr(optimizer))
          sched.step()

      # Validation phase
      result = evaluate(model, val_loader)
      result['train_loss'] = torch.stack(train_losses).mean().item()
      result['lrs'] = lrs
      model.epoch_end(epoch, result)
      history.append(result)
  return history

layers_types = [
  [1,1,1,1], # minimum
  [2,1,1,1],
  [1,2,1,1],
  [1,1,2,1],
  [1,1,1,2],
  [2,2,2,2], # resnet18
  [3,3,3,3], # intermediate
  [3,4,6,3] # resnet34
]

block_types = [BasicBlock, Bottleneck]

if __name__ == "__main__":
  # Data transforms (normalization & data augmentation)
  stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                          tt.RandomHorizontalFlip(),
                          tt.ToTensor(),
                          tt.Normalize(*stats,inplace=True)])
  valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

  batch_size = 400

  # PyTorch data loaders
  train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
  valid_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
  # PyTorch data loaders
  train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
  valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)




  device = get_default_device()
  print("using device:", device)
  train_dl = DeviceDataLoader(train_dl, device)
  valid_dl = DeviceDataLoader(valid_dl, device)

  all_models = [CIFAR_ResNet_Custom(3, 10, block, layers) for block in block_types for layers in layers_types]
  torch.cuda.empty_cache()
  all_models = [to_device(one_model, device) for one_model in all_models]

  all_history = [[evaluate(one_model, valid_dl)] for one_model in all_models]

  epochs = 100
  max_lr = 0.01
  grad_clip = 0.1
  weight_decay = 1e-4
  opt_func = torch.optim.Adam

  for idx in range(len(all_models)):
    all_models[idx]
    all_history[idx] += fit_one_cycle(epochs, max_lr, all_models[idx], train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)
    df = pd.DataFrame(all_history[idx])
    df.to_csv('history-'+str(idx)+'.csv')
