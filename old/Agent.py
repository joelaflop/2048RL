import random
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import namedtuple


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

'''import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter'''
# %load_ext tensorboard

from torch import save
from torch import load
from torch.autograd import Variable
from torch.distributions import Categorical
import os
from os import path
from WebDriver import myWebDriver
import HelperBase2 as h

def load_model(name):
    r = Net()
    r.load_state_dict(load(path.join(path.abspath(''), name), map_location='cpu'))
    return r

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input channels = 1, output channels = 20
        self.convP1 = torch.nn.Conv2d(16, 256, kernel_size=3, stride=1, padding=1)
        self.convP2 = torch.nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1)
        self.convP1_drop = nn.Dropout2d(.3)
        self.convP2_drop = nn.Dropout2d(.25)
        self.fcP1 = torch.nn.Linear(1024, 4)
        self.fcP1_drop = nn.Dropout2d(.2)

        self.convV1 = torch.nn.Conv2d(16, 256, kernel_size=3, stride=1, padding=1)
        self.convV2 = torch.nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=1)
        self.convV1_drop = nn.Dropout2d(.2)
        self.convV2_drop = nn.Dropout2d(.15)
        self.fcV1 = torch.nn.Linear(1024, 1)

        self.distribution = torch.distributions.Categorical
        self.epsilon = 1e-7

    def forward(self, x):
        p = self.convP1(x)
        #print('conv1 shape: ' + str(x.shape))
        p = self.convP1_drop(p)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (20, 64, 64) to (30, 32, 32)
        p = F.max_pool2d(p, kernel_size=2)
        #print('pool1 shape: ' + str(x.shape))
        p = F.prelu(p, torch.tensor(.25))
        #Size changes from (20, 32, 32) to (30, 32, 32)
        p = self.convP2(p)
        #print('conv2 shape: ' + str(x.shape))
        p = self.convP2_drop(p)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        p = F.max_pool2d(p, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        p = F.prelu(p, torch.tensor(.25))
        p = p.view((-1, 1024))
        #print('post view shape: ' + str(x.shape))
        p = self.fcP1(p)


        v = self.convV1(x)
        #print('conv1 shape: ' + str(x.shape))
        v = self.convV1_drop(v)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (20, 64, 64) to (30, 32, 32)
        v = F.max_pool2d(v, kernel_size=2)
        #print('pool1 shape: ' + str(x.shape))
        v = F.prelu(v, torch.tensor(.25))
        #Size changes from (20, 32, 32) to (30, 32, 32)
        v = self.convV2(v)
        #print('conv2 shape: ' + str(x.shape))
        v = self.convV2_drop(v)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        v = F.max_pool2d(v, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        v = F.prelu(v, torch.tensor(.25))
        v = v.view((-1, 1024))
        #print('post view shape: ' + str(x.shape))
        v = self.fcV1(v)


        return p, v



class myAgent():
    def __init__ (self):
        print('agent open')


    def reset(self):
        self.observation = h.newBoard()
        self.steps = 0
        return self.observation



    def step(self, d):
        qstates, nstates = h.randomNewStates(self.observation)
        newBoard = random.choice(nstates[d])

        done = (h.boardEquals(newBoard, self.observation))

        reward = h.mergeMaxReward(newBoard, self.observation)

        info = h.merges(newBoard, self.observation)

        self.observation = newBoard

        self.steps += 1
        return self.observation, reward, done, info

    def close(self):
        print('agent closed')

def power2mat3d(a):
    a[a == 0] = 1
    a = np.log2(a)

    b = np.zeros(shape=(16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
              b[int(a[i, j]), i, j] = 1.0
    return b


def boardEquals(a, b):
    return (a.astype(int) == b.astype(int)).all()

def moveLeftAndCombine(a):
    valid_mask = a != 0
    flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(a.shape[1]-1,-1,-1)
    flipped_mask = flipped_mask[:,::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0

    for r in range(4):
        for c in range(3):
            if a[r, c] > 0 and a[r, c] == a[r, c+1]:
                a[r, c] += 1
                a[r, c+1] = 0

    valid_mask = a != 0
    flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(a.shape[1]-1,-1,-1)
    flipped_mask = flipped_mask[:,::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0

    return a

class webAgent():
    def __init__ (self):
        self.brain = load_model('2048net.th')
        self.driver = myWebDriver()


    def reset(self):
        self.observation = self.driver.getBoard()
        return self.observation



    def selectBestAction(self, board):
        i = 3
        logits, _ = self.brain.forward(torch.FloatTensor(power2mat3d(board).reshape((1,16,4,4))))
        output = F.softmax(logits, dim = 1)
        print("probabilities: {}".format(output))
        actions = torch.argsort(output.data, 1)
        action = actions[0,i].item()

        u = np.copy(board.T)
        d = np.flip(np.copy(board.T), axis = 1)
        r = np.flip(np.copy(board), axis = 1)
        l = np.copy(board)

        u = moveLeftAndCombine(u).T
        d = np.flip(moveLeftAndCombine(d), axis = 1).T
        r = np.flip(moveLeftAndCombine(r), axis = 1)
        l = moveLeftAndCombine(l)

        qstates = [u,d,r,l]

        while(boardEquals(board, qstates[action])):
          i -= 1
          action = actions[0,i].item()
          if(i<0) or output[0,i] == 0:
            break
        print("chosen action: {}".format(action))
        return action

    def step(self, d):
        oldScore, newScore = self.driver.move(d)

        done = False

        info = self.driver.getInfo()

        self.observation = self.driver.getBoard()

        return self.observation, newScore, done, info

    def close(self):
        self.driver.close()
