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
from copy import deepcopy, copy

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

def average_models(model, clients_models_hist:list , weights:list):


    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)

    return new_model

def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, opt_func, mu=0,
    file_name="test", epochs=5, max_lr=10**-2, weight_decay=1e-4, grad_clip=None):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f=loss_classifier

    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    print("Clients' weights:",weights)


    loss_hist=[[float(loss_dataset(model, dl, loss_f).detach())
        for dl in training_sets]]
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist=[[tens_param.detach().cpu().numpy()
        for tens_param in list(model.parameters())]]
    models_hist = []


    server_losses = []
    server_accs = []
    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    server_losses.append(server_loss)
    server_accs.append(server_acc)

    for i in range(n_iter):

        clients_params=[]
        clients_models=[]
        clients_losses=[]

        for k in range(K):

            local_model=deepcopy(model)
            # local_optimizer=optim.SGD(local_model.parameters(),max_lr=lr)
            # Set up cutom optimizer with weight decay
            local_optimizer = opt_func(local_model.parameters(), max_lr, weight_decay=weight_decay)
            # Set up one-cycle learning rate scheduler
            # print("\nDEBUG", k, len(testing_sets[k]), "EPOCH=", epochs)
            sched = torch.optim.lr_scheduler.OneCycleLR(local_optimizer, max_lr, epochs=epochs,
                                                        steps_per_epoch=len(testing_sets[k].dataset)) # KC: DEBUG

            local_loss=local_learning(local_model,mu,local_optimizer,
                training_sets[k],epochs,loss_f, sched, grad_clip)

            clients_losses.append(local_loss)

            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))


        #CREATE THE NEW GLOBAL MODEL
        model = average_models(deepcopy(model), clients_params,
            weights=weights)
        models_hist.append(clients_models)

        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach())
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        server_losses.append(server_loss)
        server_accs.append(server_acc)


        server_hist.append([tens_param.detach().cpu().numpy()
            for tens_param in list(model.parameters())])

        # #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        # lr*=decay

    return model, server_losses, server_accs, loss_hist, acc_hist

def loss_classifier(predictions,labels):

    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")

    return loss(m(predictions) ,labels.view(-1))


def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0

    device = get_default_device()
    # dataset = dataset.to(device)

    for idx,(features,labels) in enumerate(dataset):

        features = features.to(device)
        labels = labels.to(device)

        predictions= model(features)
        loss+=loss_f(predictions,labels)

    loss/=idx+1
    return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""

    correct=0
    device = get_default_device()

    num_data = 0

    for features,labels in iter(dataset):

        num_data += len(features)
        features = features.to(device)
        labels = labels.to(device)

        predictions= model(features)

        _,predicted=predictions.max(1,keepdim=True)

        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()

    accuracy = 100*correct/num_data

    return accuracy


def train_step(model, model_0, mu:int, optimizer, train_data, loss_f, sched, grad_clip=None):
    """Train `model` on one epoch of `train_data`"""

    total_loss=0
    device = get_default_device()
    # print(len(train_data))

    for idx, (features,labels) in enumerate(train_data):

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions= model(features)

        loss=loss_f(predictions,labels)
        loss+=mu/2*difference_models_norm_2(model,model_0)
        total_loss+=loss

        loss.backward()

        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        sched.step()
        # print("STEP\n")

    return total_loss/(idx+1)



def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f, sched, grad_clip=None):

    model_0=deepcopy(model)

    for e in range(epochs):
        local_loss=train_step(model,model_0,mu,optimizer,train_data,loss_f, sched, grad_clip)

    return float(local_loss.detach().cpu().numpy())


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """

    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())

    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2)
        for i in range(len(tensor_1))])

    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert(nb_nodes>0 and nb_nodes<=10)

    digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    # split the digits in a fair way
    digits_split=list()
    i=0
    for n in range(nb_nodes, 0, -1):
        inc=int((10-i)/n)
        digits_split.append(digits[i:i+inc])
        i+=inc

    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=nb_nodes*n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = next(dataiter)

    data_splitted=list()
    for i in range(nb_nodes):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))

    return data_splitted



def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)

    data_splitted=list()
    for _ in range(nb_nodes):
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(next(dataiter))), batch_size=batch_size, shuffle=shuffle))

    return data_splitted

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
  # train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
  # valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)
  n_samples_train = 2000
  n_samples_test = 1000
  train_dls=iid_split(train_ds, 10, n_samples_train, batch_size, shuffle=True)
  valid_dls=iid_split(valid_ds, 10, n_samples_test, batch_size*2, shuffle=False)





  device = get_default_device()
  print("using device:", device)
  # train_dls = DeviceDataLoader(train_dls, device)
  # valid_dls = DeviceDataLoader(valid_dls, device)

  all_models = [CIFAR_ResNet_Custom(3, 10, block, layers) for block in block_types for layers in layers_types]
  torch.cuda.empty_cache()
  all_models = [to_device(one_model, device) for one_model in all_models]

  # all_history = [[evaluate(one_model, valid_dl)] for one_model in all_models]

  # epochs = 100
  max_lr = 0.01
  grad_clip = 0.1
  weight_decay = 1e-4
  opt_func = torch.optim.Adam
  n_iter = 200 # comm rounds
  mu = 0 # 0 for fedavg

  for idx in range(len(all_models)):
    # all_models[idx]
    # all_history[idx] += fit_one_cycle(epochs, max_lr, all_models[idx], train_dl, valid_dl,
    #                          grad_clip=grad_clip,
    #                          weight_decay=weight_decay,
    #                          opt_func=opt_func)

    model_f, server_losses, server_accs, loss_hist_FA_iid, acc_hist_FA_iid = FedProx(
      all_models[idx],
      train_dls, n_iter, valid_dls, opt_func, epochs=2,
      max_lr=max_lr, mu=mu, weight_decay=weight_decay, grad_clip=grad_clip
    )
    data_to_save = np.array([server_losses, server_accs, loss_hist_FA_iid, acc_hist_FA_iid]).T
    df = pd.DataFrame(data_to_save, columns=["glob_loss", "glob_acc", "local_loss", "local_acc"])
    df.to_csv('iid-history-'+str(idx)+'.csv')
