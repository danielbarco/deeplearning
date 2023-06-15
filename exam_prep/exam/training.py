import torch


#########################
# Training Loop 
#########################


# Train convolutional & linear

def train(network, epochs = 1000, eta = 0.01, momentum = 0.9):
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
    for x,t in train_loader:
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
      for x,t in test_loader:
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


