import torch
import tqdm
import math


#########################
# Utility functions
#########################

def standardize(X_train, X_val):
  # compute statistics
  ## ??? why dim = 0 ???
  # mean per column i.e. mean per parameter
  mean = torch.mean(X_train, dim=0)
  std = torch.std(X_train, dim=0)

  # standardize both X_train and X_val
  X_train = (X_train - mean) / std
  X_val = (X_val - mean) / std
  return X_train, X_val

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
    Y = (Z >= 0).float()
    return torch.mean((Y == T).float())

  else:
    # categorical classification
    # the argmax function returns the index of the maximum value
    # we use the .float() function to convert the boolean to a float
    # then we compare the prediction with the target and compute the mean
    # return torch.mean((torch.argmax(Z, dim=1).float() == T).float())
    
    # Y is the index of the maximum value in Z
    Y = torch.argmax(Z, dim=1)
    return torch.mean((Y == T).float())
  
def output_dim(features_in, kernel_size, padding, stride, dilation = 1):
    """returns the output dimension of a convolutional network 
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    (floor rounds down to nearest int)
    """
    return math.floor((features_in + 2 * padding - dilation * (kernel_size - 1) -1) / stride ) +1
  
print('first layer output dimensions: ', 
      output_dim(features_in = 28 , kernel_size = 5, padding = 2, stride = 1))
print('first maxpooling output dimensions: ', 
      output_dim(features_in = 28 , kernel_size = 2, padding = 0, stride = 2))
print('second layer output dimensions: ', 
      output_dim(features_in = 14 , kernel_size = 5, padding = 2, stride = 1))
print('second maxpooling output dimensions: ', 
      output_dim(features_in = 14 , kernel_size = 2, padding = 0, stride = 2))

#########################
# Network Definition
#########################

# Linear network
# D: input dimension
# K: hidden dimension
# O: output dimension

def Network(D, K, O):
  return torch.nn.Sequential(
    torch.nn.Linear(D, K),
    torch.nn.Tanh(),
    torch.nn.Linear(K, O),
    
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
 
# Convolutional network
# Q1: number of filters in the first convolutional layer
# Q2: number of filters in the second convolutional layer
# O: output dimension

def convolutional(Q1, Q2, O):
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
  
#########################
# Network Module Convolutions
#########################

class Network (torch.nn.Module):
  def __init__(self, Q1, Q2, K, O):
    # call base class constrcutor
    super(Network,self).__init__()
    # define convolutional layers
    self.conv1 = torch.nn.Conv2d(1, Q1, kernel_size= [5,5], stride=1, padding=2)
    self.conv2 = torch.nn.Conv2d(Q1, Q2, kernel_size= [5,5], stride=1, padding=2)
    # pooling and activation functions will be re-used for the different stages
    self.pool = torch.nn.MaxPool2d(kernel_size=[2,2], stride=2)
    self.act = torch.nn.ReLU()
    # define fully-connected layers
    self.flatten = torch.nn.Flatten()
    # ?? how do I know the size of the input to the first fully-connected layer?
    
    self.fc1 = torch.nn.Linear(Q2 * 7 * 7, K)
    self.fc2 = torch.nn.Linear(K, O)
  
  def forward(self,x):
    # compute first layer of convolution, pooling and activation
    a = self.act(self.pool(self.conv1(x)))
    # compute second layer of convolution, pooling and activation
    a = self.act(self.pool(self.conv2(a)))
    # get the deep features as the output of the first fully-connected layer
    # ?? why do I flatten before input to the first fully-connected layer?
    deep_features = self.act(self.fc1(self.flatten(a)))
    # get the logits as the output of the second fully-connected layer
    logits = self.fc2(deep_features)
    # return both the logits and the deep features
    return logits, deep_features

# run on cuda device
device = torch.device("cuda")
# create network with 20 hidden neurons in FC layer
network = Network(Q1=16, Q2=32, K= 20, O= 4).to(device)

#########################
# Autograd
#########################

from torch._C import wait
class MyFunction(torch.autograd.Function):

  # implement the forward propagation
  @staticmethod
  def forward(ctx, x, w):
    # compute the output
    output = torch.sum(x-w, dim=0)**2
    # save required parameters for backward pass
    ctx.save_for_backward(x, w)
    return output

  # implement Jacobian
  @staticmethod
  def backward(ctx, grad):
    # get results stored from forward pass
    x, w = ctx.saved_tensors
    # compute the derivatives
    # chain_rule = outer'(inner) * inner'
    # outer'(inner) = 2 * (x-w)
    # inner' = -1
    dJ_dw = -2 * (x-w)
    return dJ_dw, None #because we do not need the da/dx -> None


#########################
# Custom Network Layer
#########################

class RBFLayer(torch.nn.Module):
  def __init__(self, K, R):
    # call base class constructor
    super(RBFLayer, self).__init__()
    self.K = K
    self.R = R
    # store a parameter for the basis functions
    self.W = torch.nn.Parameter()
    # initialize the matrix between -2 and 2
    self.W.data.uniform_(-2, 2)

  def forward(self, x):
    # collect the required shape parameters, B, R, K
    # B, R, K = Batch, x.shape[1], class
    B, R, K = x.shape[0], self.R, self.K
    print(B,R,K)
    # Bring the weight matrix of shape R,K to size B,R,K by adding batch dimension (B, dim 0)
    W = self.W.expand(B)
    print(W.shape)
    # Bring the input matrix of shape B,K to size B,R,K by adding R dimension (dim=1)
    X = x.expand(B, R, K)
    # compute the activation euclidean distance
    A = torch.sqrt(torch.sum((W - X)**2, dim=2))
    return A
  
class RBFActivation(torch.nn.Module):
  def __init__(self, R):
    # call base class constructor
    super(RBFActivation, self).__init__()
    self.R = R
    # store a parameter for the basis functions
    sigma = torch.nn.Parameter()
    # initialize sigma 1
    sigma.data.fill_(1)
    self.sigma2 = 2 * sigma**2
  def forward(self, x):
    # implement the RBF activation function
    output = torch.exp(- x**2 / self.sigma2)
    return output

