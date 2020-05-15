import random
import numpy as np
import pandas as pd
import math
from collections import namedtuple


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch import save
from torch import load
from torch.autograd import Variable
from torch.distributions import Categorical
from os import path

from WebDriver import myWebDriver
import HelperBase2 as h



def power2mat3d(a):
    b = np.zeros(shape=(16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
              b[int(a[i, j]), i, j] = 1.0
    return b

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
        #p = self.convP1_drop(p)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (20, 64, 64) to (30, 32, 32)
        p = F.max_pool2d(p, kernel_size=2)
        #print('pool1 shape: ' + str(x.shape))
        #p = F.prelu(p, torch.tensor(.25))
        p = F.relu(p)
        #Size changes from (20, 32, 32) to (30, 32, 32)
        p = self.convP2(p)
        #print('conv2 shape: ' + str(x.shape))
        #p = self.convP2_drop(p)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        p = F.max_pool2d(p, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        #p = F.prelu(p, torch.tensor(.25))
        p = F.relu(p)
        p = p.view((-1, 1024))
        #print('post view shape: ' + str(x.shape))
        p = self.fcP1(p)


        v = self.convV1(x)
        #print('conv1 shape: ' + str(x.shape))
        #v = self.convV1_drop(v)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (20, 64, 64) to (30, 32, 32)
        v = F.max_pool2d(v, kernel_size=2)
        #print('pool1 shape: ' + str(x.shape))
        #v = F.prelu(v, torch.tensor(.25))
        v = F.relu(v)
        #Size changes from (20, 32, 32) to (30, 32, 32)
        v = self.convV2(v)
        #print('conv2 shape: ' + str(x.shape))
        #v = self.convV2_drop(v)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        v = F.max_pool2d(v, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        #v = F.prelu(v, torch.tensor(.25))
        v = F.relu(v)
        v = v.view((-1, 1024))
        #print('post view shape: ' + str(x.shape))
        v = self.fcV1(v)


        return p, v

    def loss_func(self, states, actions, values_target):
        self.train()
        logits, values = self.forward(states)
        values_difference = values_target - values
        c_loss = values_difference ** 2

        probs = F.softmax(logits, dim=1)
        '''probs = F.relu(probs)
        probs = probs + 1e-10'''
        m = self.distribution(probs)
        prob_entropy = m.entropy()
        exp_v = m.log_prob(actions) * values_difference.detach().squeeze()
        a_loss = -exp_v + prob_entropy
        return c_loss + a_loss

    #def choose_action(self, s): has to be in Agent class to acocunt for non-moves

def runChoice(p, episodes):
    #running_reward = 10
    agent = myAgent()

    p.eval()
    totTiles = np.zeros(13)
    donecount = 0
    game_rewards = []

    ep = 0
    while ep < episodes:
        print('\rEpisode: '+str(ep + 1), end="")
        state = agent.reset()
        done = False
        rewards = []

        while done == False:
            s = np.empty((1,16,4,4))
            s[0,:,:,:] = power2mat3d(state)

            with torch.no_grad():
              logits, val = p(torch.from_numpy(s).type(torch.FloatTensor))
            probs = F.softmax(logits, dim = 1)
            action = agent.selectChoiceAction(probs)

            #action = agent.selectValueAction(p)

            state, reward, done, _ = agent.step(action)
            rewards.append(reward)

            if done:
                #print(state)
                ep += 1
                bestTile = int(np.max(state))
                totTiles[bestTile-1] += 1
                donecount += 1
                game_rewards.append(sum(rewards))
                #print(sum(rewards))
                #print('Game reward:{} over {} moves'.format(sum(rewards), len(rewards)))
                #print(state)
                break
    print("tile distr: "+str(totTiles))
    print("avg tile: "+str(np.dot(totTiles, np.arange(1,14))/np.sum(totTiles)))
    print('average reward: {}'.format(sum(game_rewards)/episodes))
    print((donecount == episodes))


class myAgent():
    def __init__ (self, driver):
        print('open')
        self.qstates = []
        self.nstates = []
        self.driver = driver

    def reset(self):
        self.observation = self.driver.getlgBoard() #newBoard()
        self.steps = 0
        self.qstates, self.nstates = randomNewStates(self.observation)
        return self.observation

    def step(self, d):
        newBoard = self.driver.getlgBoard() # random.choice(self.nstates[d])

        '''done = ((self.observation.astype(int) == qstates[0].astype(int)).all() and
                (self.observation.astype(int) == qstates[1].astype(int)).all() and
                (self.observation.astype(int) == qstates[2].astype(int)).all() and
                (self.observation.astype(int) == qstates[3].astype(int)).all())'''

        done = (self.observation.astype(int) == self.qstates[d].astype(int)).all()

        #reward = mergeMaxReward(newBoard, self.observation)  #possiblemergesreward(newBoard, self.observation) #newReward(newBoard, self.observation)

        reward = reward2048(newBoard, self.observation)

        self.observation = newBoard

        self.qstates, self.nstates = randomNewStates(self.observation)

        info = 'no info'

        self.steps += 1
        return self.observation, reward, done, info
    def selectValueAction(self, net):
        s_up = np.empty((1,16,4,4))
        s_up[0,:,:,:] = power2mat3d(self.qstates[0])
        _, upval = p(s_up)

        s_down = np.empty((1,16,4,4))
        s_down[0,:,:,:] = power2mat3d(self.qstates[1])
        _, downval = p(s_down)

        s_right = np.empty((1,16,4,4))
        s_right[0,:,:,:] = power2mat3d(self.qstates[2])
        _, rightval = p(s_right)

        s_left = np.empty((1,16,4,4))
        s_left[0,:,:,:] = power2mat3d(self.qstates[3])
        _, leftval = p(s_left)

        return np.argmax(np.array([upval, downval, rightval, leftval]))

    def selectBestAction(self, output):
        i = 3
        actions = torch.argsort(output.data, 1)
        action = actions[0,i].item()

        while(boardEquals(self.observation, self.qstates[action])):
          i -= 1
          action = actions[0,i].item()
          if(i<0) or output[0,i] == 0:
            break
        return action

    def selectRandomAction(self, output):
        action_space = np.arange(4)
        action_probs = output.detach().cpu().numpy()[0]
        action = random.randint(0,3) #np.random.choice(action_space, p=action_probs)

        deleted = [False, False, False, False]
        while(boardEquals(self.observation, self.qstates[action])):

          deleted[action] = True
          deletion = action_probs[action]
          action_probs[action] = 0
          action_probs[~np.array(deleted)] += deletion/(len(action_probs) - np.sum(deleted))

          action = random.randint(0,3) #np.random.choice(action_space, p=action_probs)

          if(np.sum(deleted) == 3):
            break
        return action

    def selectChoiceAction(self, output):
        action_space = np.arange(4)
        action_probs = output.detach().cpu().numpy()[0]
        action = np.random.choice(action_space, p=action_probs)

        deleted = [False, False, False, False]
        while(boardEquals(self.observation, self.qstates[action])):

          deleted[action] = True
          deletion = action_probs[action]
          action_probs[action] = 0
          action_probs[~np.array(deleted)] += deletion/(len(action_probs) - np.sum(deleted))

          action = np.random.choice(action_space, p=action_probs)

          if(np.sum(deleted) == 3):
            break
        return action

    def close(self):
        print('done')


'''
helper functions
'''

def newBoard():
    b = np.zeros((4,4))

    r = random.randint(0,3)
    c = random.randint(0,3)

    rr = random.randint(0,3)
    cc = random.randint(0,3)

    while rr == r and cc == c:
        rr = random.randint(0,3)
        cc = random.randint(0,3)

    b[r, c] = 1 if random.random() < .9 else 2
    b[rr, cc] = 1 if random.random() < .9 else 2

    return b

weights = np.array([[16,8,4,2],
                    [2**11,2**10,2**9,2**8],
                    [2**18,2**17,2**16,2**15],
                    [2**27,2**28,2**29,2**30]])
new_weights = np.flip(np.array([0,2**0,2**1,2**2, 2**3,2**4,2**5,2**6, 2**7,2**8,2**9,2**10, 2**16,2**17,2**18,2**19]))
newer_weights = np.array([0, 0, 1, 2,  3, 4, 5, 6,  7, 8, 9, 10,  16, 17, 18, 19])

def manualreward(new, old):
    reward = 0
    nonmoves = 0
    for r in range(4):
        for c in range(4):
            if(new[r,c] == old[r,c]):
                nonmoves += 1
            reward += (new[r,c])*weights[r,c] # 0 if new[r,c] == 0 else math.log(new[r,c],2)
    return reward  if nonmoves < 16 else 0 #- 10*unsortedness(a)

def reward2048(new, old):
    return .01 if np.max(new) > 10 else 0

def mergeMaxReward(new, old):
    return 0 if np.equal(new, old).all() else ((merges(new, old)-5)/10 + np.max(new)*.1)

def possiblemergesreward(new, old):
    return (possiblemerges(new) - possiblemerges(old))/6

def possiblemerges(b):
    m = 0
    for r in range(4):
        for c in range(4):
            if r < 3:
                if b[r,c] == b[r+1,c] and b[r,c] != 0:
                    m += 1
            if c < 3:
                if b[r,c] == b[r,c+1] and b[r,c] != 0:
                    m += 1
    return m

def merges(new, old):
    newE = 0
    oldE = 0
    for r in range(4):
        for c in range(4):
            if(new[r,c] == 0):
                newE += 1
            if(old[r,c] == 0):
                oldE += 1
    return (newE-oldE + 1)


def unsortedness(a):
    reward = 0
    zig = zigzagOrder(np.copy(a))
    sorted = np.sort(zigzagOrder(np.copy(a)))
    wrongs = np.zeros(16)

    distance = 0
    for i in range (16):
        if zig[i] != sorted [i]:
            distance += 1
            wrongs[i] = 1

    difference = np.dot(wrongs, new_weights)

    '''print(zig)
    print(sorted)
    print(wrongs)'''

    reward = np.sum(difference)

    return reward

def worstNewStates(a):
    u = np.copy(a.T)
    d = np.flip(np.copy(a.T), axis = 1)
    r = np.flip(np.copy(a), axis = 1)
    l = np.copy(a)

    u = moveLeftAndCombine(u).T
    d = np.flip(moveLeftAndCombine(d), axis = 1).T
    r = np.flip(moveLeftAndCombine(r), axis = 1)
    l = moveLeftAndCombine(l)

    uu = [np.copy(u)] if boardEquals(a, u) else addWorstSpawn(np.copy(u))
    dd = [np.copy(d)] if boardEquals(a, d) else addWorstSpawn(np.copy(d))
    rr = [np.copy(r)] if boardEquals(a, r) else addWorstSpawn(np.copy(r))
    ll = [np.copy(l)] if boardEquals(a, l) else addWorstSpawn(np.copy(l))

    return [u,d,r,l], [uu, dd, rr, ll]

def randomNewStates(a):
    u = np.copy(a.T)
    d = np.flip(np.copy(a.T), axis = 1)
    r = np.flip(np.copy(a), axis = 1)
    l = np.copy(a)

    u = moveLeftAndCombine(u).T
    d = np.flip(moveLeftAndCombine(d), axis = 1).T
    r = np.flip(moveLeftAndCombine(r), axis = 1)
    l = moveLeftAndCombine(l)

    uu = [np.copy(u)] if boardEquals(a, u) else addRandomSpawns(np.copy(u))
    dd = [np.copy(d)] if boardEquals(a, d) else addRandomSpawns(np.copy(d))
    rr = [np.copy(r)] if boardEquals(a, r) else addRandomSpawns(np.copy(r))
    ll = [np.copy(l)] if boardEquals(a, l) else addRandomSpawns(np.copy(l))

    return [u,d,r,l], [uu, dd, rr, ll]

def addRandomSpawns(a):
    zlist = []
    for r in range(4):
        for c in range(4):
            if a[r, c] == 0:
                zlist.append((r,c))
    if zlist:
        alist = []
        for spawn in zlist:
            a2 = np.copy(a)
            a2[spawn] = 1
            a4 = np.copy(a)
            a4[spawn] = 2
            for i in range(9):
                alist.append(a2)
            alist.append(a4)
    else:
        alist = [a]
    return alist

def addWorstSpawn(a):
    zlist = []
    for r in range(4):
        for c in range(4):
            if a[r, c] == 0:
                zlist.append((r,c))
    if zlist:
        alist = []
        maxr = zlist[0][0]
        maxc = zlist[0][1]
        for r,c in zlist:
            if(r >= maxr):
                maxr = r
        for r,c in zlist:
            if(r == maxr and c >= maxc):
                maxc = c
        aa = np.copy(a)
        aa[(maxr, maxc)] = 1
        alist = [aa]
    else:
        alist = [a]
    return alist

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

def boardEquals(a, b):
    return (a.astype(int) == b.astype(int)).all()

def zigzagOrder(a):
    return np.concatenate((np.flip(a[0]), a[1], np.flip(a[2]), a[3]))

import time

def run(p, episodes, driver):
    #running_reward = 10
    agent = myAgent(driver)

    p.eval()
    totTiles = np.zeros(13)
    donecount = 0
    game_rewards = []

    ep = 0
    while ep < episodes:
        print('\rEpisode: '+str(ep + 1), end="")
        state = agent.reset()
        done = False
        rewards = []
        time.sleep(10)

        while done == False:
            s = np.empty((1,16,4,4))
            s[0,:,:,:] = power2mat3d(state)

            with torch.no_grad():
              logits, val = p(torch.from_numpy(s).type(torch.FloatTensor))
            probs = F.softmax(logits, dim = 1)
            action = agent.selectBestAction(probs)
            driver.move(action)
            time.sleep(.05)

            #action = agent.selectValueAction(p)

            state, reward, done, _ = agent.step(action)
            rewards.append(reward)

            if done:
                #print(state)
                ep += 1
                bestTile = int(np.max(state))
                totTiles[bestTile-1] += 1
                donecount += 1
                game_rewards.append(sum(rewards))
                #print(sum(rewards))
                #print('Game reward:{} over {} moves'.format(sum(rewards), len(rewards)))
                #print(state)
                break
    print("tile distr: "+str(totTiles))
    print("avg tile: "+str(np.dot(totTiles, np.arange(1,14))/np.sum(totTiles)))
    print('average reward: {}'.format(sum(game_rewards)/episodes))
    print((donecount == episodes))

def do():
    brain = load_model('good.th')
    driver = myWebDriver()
    run(brain, 10, driver)

do()
