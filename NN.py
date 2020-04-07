import random
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


def power2mat3d(a):
    b = np.zeros(shape=(1, 4,4,16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(a[i, j]==0):
                b[0, i, j, 0] = 1.0
            else:
                b[0, i, j, int(a[i, j])] = 1.0
    return b

def power2mat(a):
    b = np.zeros(shape=(1,4,4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if(a[i, j]==0):
                b[0, i, j] = 0
            else:
                b[0, i, j] = a[i, j]
    return b

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input channels = 1, output channels = 20
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_drop = nn.Dropout2d(.35)
        self.conv2_drop = nn.Dropout2d(.5)

        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #XXXX input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(512, 256)

        #64 input features, 6 output features for our 6 defined classes
        self.fc2 = torch.nn.Linear(256, 4)

    def forward(self, x):
        print('FORWARD START')
        print('start shape: ' + str(x.shape))
        #Size changes from (1, 64, 64) to (20, 64, 64)
        x = self.conv1(x)
        print('conv1 shape: ' + str(x.shape))
        x = self.conv1_drop(x)
        print('drop  shape: ' + str(x.shape))
        #Size changes from (20, 64, 64) to (30, 32, 32)
        #x = self.pool1(x)
        #print('pool1 shape: ' + str(x.shape))
        x = F.relu(x)
        print('relu  shape: ' + str(x.shape))

        #Size changes from (20, 32, 32) to (30, 32, 32)
        x = self.conv2(x)
        print('conv2 shape: ' + str(x.shape))
        x = self.conv2_drop(x)
        print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        #x = self.pool2(x)
        #print('pool2 shape: ' + str(x.shape))
        x = F.relu(x)
        print('relu  shape: ' + str(x.shape))

        #Reshape data to input to the input layer of the neural net
        #Size changes from (30, 32, 32) to (1, XXXX)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view((50, 512))
        print('post view shape: ' + str(x.shape))

        #Computes the activation of the first fully connected layer
        #Size changes from (1, XXXX) to (1, 64)
        x = F.relu(self.fc1(x))
        print('shape: ' + str(x.shape))

        x = F.dropout(x, training=self.training)
        print('shape: ' + str(x.shape))


        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 4)
        x = self.fc2(x)
        print('shape: ' + str(x.shape))
        x = F.log_softmax(x, dim =1)
        print('shape: ' + str(x.shape))
        print('FORWARD END')
        return x


def load_data(dataset_path, data_transforms=None, num_workers=0, batch_size=128):
    dataset = Dataset2048(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)


class Dataset2048(Dataset):
    def __init__(self, image_path):
        self.df = pd.read_csv(image_path+'lookaheaddata.csv')
        print(self.df.head())

    def __len__(self):
        """
        Your code here
        """
        n, d = self.df.shape
        return n

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: board, label
        """
        item = np.array(self.df.iloc[idx, 0:16]).reshape((4,4))
        item = power2mat(item)
        val = self.df.iloc[idx, 16]


        '''print(item.dtype)
        print(val.dtype)'''
        return torch.Tensor(item), val.astype(np.int64)


train_loader = load_data(dataset_path = '', batch_size = 128)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

network = Net()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

def train(epoch):
    network.train()
    stepper = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        #print(data.shape)
        optimizer.zero_grad()
        output = network(data)
        print(type(output))
        print(type(output))
        loss = F.cross_entropy(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
              train_losses.append(loss.item())

              #train_logger.add_scalar('Train loss',loss.item(),(epoch-1)*num_steps_per_epoch + stepper)

              #train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
              stepper += 1
              torch.save(network.state_dict(), '/Users/joelaflop/Documents/GitHub/2048RL/model.pth')
              torch.save(optimizer.state_dict(), '/Users/joelaflop/Documents/GitHub/2048RL/opt.pth')

train(1)
