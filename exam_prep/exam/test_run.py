
import os
import torch
import pandas as pd
import csv
from tqdm import tqdm
import torchvision
from torchvision import transforms


transform = torchvision.transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


B = 128 # batch size
# training set and data loader
trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size=B)
# validation set and data loader
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size=B)

def convolutional(Q1 = 16, Q2 =32, O=10 ):
  return torch.nn.Sequential(
      # careful in_channels = 1 not Q1
      torch.nn.Conv2d(in_channels=1, out_channels=Q1, kernel_size= (5,5), stride=1, padding=2),
      torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),
      torch.nn.Tanh(),
      torch.nn.Conv2d(in_channels=Q1, out_channels=Q2, kernel_size= (5,5), stride=1, padding=2),
      torch.nn.MaxPool2d(kernel_size=(2,2), stride=2),
      torch.nn.Tanh(),
      # flatten without parameters
      torch.nn.Flatten(),
      # ?? why 7 * 7
      torch.nn.Linear(in_features=Q2 * 7 * 7, out_features=O)
  )

 # Fully connected network
def fully_connected(D, K, O):
  return torch.nn.Sequential(
    # no parameters are required in flatten
    torch.nn.Flatten(),
    torch.nn.Linear(in_features= D, out_features=K),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features= K, out_features=K),
    torch.nn.Tanh(),
    torch.nn.Linear(in_features= K, out_features=O),

  )

def accuracy(Z, T):
  # check if we have binary or categorical classification
  # for binary classification, we will have a two-dimensional target tensor
  if len(T.shape) == 2:
    # binary classification
    # If z is equal or larger than the threshold 0.5, then we predict 1, otherwise 0 
    # we use the .float() function to convert the boolean to a float
    # then we compare the prediction with the target and compute the mean
    
    # ??? our data is binary between 0 and 1, so why do you use 0 as threshold ???
    # So only if we use sigmoid activation function for binary or softmax for multi-class
    # after the last layer, we need to use 0.5 as threshold
    # in this case we tanh as activation function, so we don't need to use 0.5 as threshold
    return torch.mean(((Z>=0).float() == T).float())

  else:
    # categorical classification
    # the argmax function returns the index of the maximum value
    # we use the .float() function to convert the boolean to a float
    # then we compare the prediction with the target and compute the mean
    # return torch.mean((torch.argmax(Z, dim=1).float() == T).float())
    
    # Y is the index of the maximum value in Z
    Y = torch.argmax(Z, dim=1)
    return torch.mean((Y == T).float())


def train(network, epochs = 10, eta = 0.01, momentum = 0.9):
  # select loss function and optimizer
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
    network.parameters(), 
    lr=eta,
    momentum= momentum,
  )

  # instantiate the correct device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  network = network.to(device)

  # collect loss values and accuracies over the training epochs
  val_loss, val_acc = [], []

  for epoch in tqdm(range(epochs)):
    # train network on training data
    for x,t in trainloader:
      # put data to device
      x, t = x.to(device), t.to(device)
      # train
      optimizer.zero_grad()
      # train on training set
      # ... compute network output on training data
      Z = network(x)
      # ... compute loss from network output and target data
      loss_ = loss(Z, t)
      loss_.backward()
      # ... perform parameter update
      optimizer.step()


    # test network on test data
    
    with torch.no_grad():
      batch_val_loss, batch_val_acc = [], []
      for x,t in testloader:
        # put data to device
        x, t = x.to(device), t.to(device)
        # compute validation loss
        Z = network(x)
        # ... compute loss from network output and target data
        loss_ = loss(Z, t)
        # ... remember loss
        batch_val_loss.append(loss_.item())
        # ... compute validation set accuracy
        batch_val_acc.append(accuracy(Z, t).item())
        
      # careful calculate loss and accuracy averaged over batches
      val_loss.append(sum(batch_val_loss) / len(batch_val_loss))
      # ... compute validation set accuracy
      val_acc.append(sum(batch_val_acc) / len(batch_val_acc))


  # return loss and accuracy values
  return val_loss, val_acc

fc = fully_connected(D=28*28, K=100, O=10 )
fc_loss, fc_acc = train(network = fc, epochs = 10, eta = 0.01, momentum = 0.9)

cv = convolutional(Q1 = 16, Q2 =32, O=10 )
cv_loss, cv_acc = train(network= cv, epochs = 10, eta = 0.01, momentum = 0.9)

from matplotlib import pyplot as plt
plt.figure(figsize=(10,3))
ax = plt.subplot(121)
# plot loss values of FC and CV network over epochs
ax.plot(fc_loss, "g-", label="Fully Connected loss")
ax.plot(cv_loss, "b-", label="Convolutional loss")
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('Validation Loss')

ax = plt.subplot(122)
# plot accuracy values of FC and CV network over epochs
ax.plot(fc_acc, "g-", label="Fully-connected accuracy")
ax.plot(cv_acc, "b-", label="Convolutional accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel('Accuracy')
ax.legend()
ax.set_title('Validation Accuracy')
plt.show()
