
import os
import torch
import pandas as pd
import csv
from tqdm import tqdm
import torchvision
from torchvision import transforms

#########################
# Wine dataset
#########################


# download the two dataset files
dataset_files = {
  "spambase.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/",
  "wine.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/"
}
for name, url in dataset_files.items():
  if not os.path.exists(name):
    import urllib.request
    urllib.request.urlretrieve(url+name, name)
    print ("Downloaded datafile", name)
    
def dataset(dataset_file="wine.data"):
  # read dataset
  data = pd.read_csv(dataset_file, header=None).values.tolist()

  print (f"Loaded dataset with {len(data)} samples")
  
  # convert to torch.tensor
  data = torch.tensor(data)

  if dataset_file == "wine.data":
    # target is in the first column and needs to be converted to long
    X = data[:, 1:]
    T = data[:, :1].flatten() - 1
    T = T.long()
  else:
    # target is in the last column and needs to be of type float
    X = data[:, :-1]
    T = data[:, -1:].float()
  return X, T

def split_training_data(X,T,train_percentage=0.8):
  # shuffle data
  ## ??? why do we need to shuffle the data here and not after every epoch ???
  ## ??? why do I get really weired results without this???
  ## data is sorted by classes
  indices = torch.randperm(len(X))
  X = X[indices]
  T = T[indices]
  
  # split into training/validation dataset
  N = X.shape[0]
  # split pytorch tensor into training and validation
  X_train = X[:int(N * train_percentage)]
  T_train = T[:int(N * train_percentage)]
  X_val = X[int(N * train_percentage):]
  T_val = T[int(N * train_percentage):]

  assert X_train.shape[0] + X_val.shape[0] == X.shape[0]
  assert T_train.shape[0] + T_val.shape[0] == T.shape[0]
  return X_train, T_train, X_val, T_val

#########################
# MINST dataset
#########################


transform = torchvision.transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
validationset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


B = 128 # batch size
# training set and data loader
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=B)

# validation set and data loader
validation_loader = torch.utils.data.DataLoader(dataset = validationset, batch_size=B)


#########################
# CIFAR10 dataset
#########################

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(
  root = "./data",
  train=True, download=True, transform=imagenet_transform
)

testset = torchvision.datasets.CIFAR10(
  root = "./data",
  train=False, download=True, transform=imagenet_transform
)

B = 64
trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size=B)
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size=B)