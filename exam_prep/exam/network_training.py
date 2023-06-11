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


#########################
# Training Loop 
#########################


# Train convolutional & linear

def train(network, epochs, eta, momentum):
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

# Train linear

def train(X_train, T_train, X_val, T_val, network, loss_function, epochs=1000, learning_rate=0.1):
  optimizer = torch.optim.SGD(
    network.parameters(), 
    lr=learning_rate,
  )
  X_train.requires_grad = True
  # collect loss and accuracy values
  train_loss, train_acc, val_loss, val_acc = [], [], [], []

  for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()

    # train on training set
    # ... compute network output on training data
    Z = network(X_train)
    # ... compute loss from network output and target data
    loss = loss_function(Z, T_train)
    loss.backward()
    # ... perform parameter update
    optimizer.step()
    # ... remember loss
    train_loss.append(loss.item())
    # ... compute training set accuracy
    train_acc.append(accuracy(Z, T_train))

    # test on validation data
    with torch.no_grad():
      # ... compute network output on validation data
      Z = network(X_val)
      # ... compute loss from network output and target data
      loss = loss_function(Z, T_val)
      # ... remember loss
      val_loss.append(loss.item())
      # ... compute validation set accuracy
      val_acc.append(accuracy(Z, T_val).item())

  # return the four lists of losses and accuracies
  return train_loss, train_acc, val_loss, val_acc




#########################
# Training
#########################

# Binary classification


# # define loss function
loss = torch.nn.BCEWithLogitsLoss()
# load dataset
X, T = dataset("spambase.data")
# split dataset
X_train, T_train, X_val, T_val = split_training_data(X,T)
# standardize input data
X_train, X_val = standardize(X_train, X_val)
# instantiate network
network = Network(X.shape[1], K = 10, O = 1)

# train network on our data
results = train(X_train, T_train, X_val, T_val, network, loss, epochs=10000, learning_rate=0.1)

# Categorical classification

# define loss function
loss = torch.nn.CrossEntropyLoss()
# load dataset
X, T = dataset("wine.data")
# split dataset
X_train, T_train, X_val, T_val = split_training_data(X,T)
# standardize input data
X_train, X_val = standardize(X_train, X_val)
# instantiate network
network = Network(X.shape[1], K = 10, O = 3)

# train network on our data
results = train(X_train, T_train, X_val, T_val, network, loss, epochs=10000, learning_rate=0.1)


