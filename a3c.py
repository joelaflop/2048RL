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

#if not torch.cuda.is_available():
#  print("no GPU")
#dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#dev = torch.device("cpu")

# Commented out IPython magic to ensure Python compatibility.
#going to have to mount your own drive directory
'''from google.colab import drive
drive.mount('/content/drive', force_remount=True)'''
# %cd '/content/drive/My Drive/bollege/year three/cis419/final'

gamma = .99
update_global = 64
max_eps = 2000000

os.environ["OMP_NUM_THREADS"] = "1"

"""# Data helpers"""

def power2mat3d(a):
    b = np.zeros(shape=(16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
              b[int(a[i, j]), i, j] = 1.0
    return b

def save_model(model, name):
    if isinstance(model, Net):
        return save(model.state_dict(), path.join(path.abspath(''), name))
    #return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))

def load_model(name):
    r = Net()
    r.load_state_dict(load(path.join(path.abspath(''), name), map_location='cpu'))
    return r


"""#Agent Class and 2048 helpers"""

class myAgent():
    def __init__ (self):
        print('open')
        self.qstates = []
        self.nstates = []

    def reset(self):
        self.observation = newBoard()
        self.steps = 0
        self.qstates, self.nstates = bestNewStates(self.observation)
        return self.observation

    def step(self, d):
        newBoard = random.choice(self.nstates[d])

        done = (self.observation.astype(int) == self.qstates[d].astype(int)).all()# or np.max(newBoard) == 10

        #reward = mergeMaxReward(newBoard, self.observation)
        #reward = reward2048(newBoard, self.observation)
        reward = rewardMaxTileSplits(newBoard, self.observation, done)

        self.observation = newBoard

        self.qstates, self.nstates = bestNewStates(self.observation)

        info = np.max(self.observation)

        self.steps += 1
        return self.observation, reward, done, info

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

def reward2048(new, old, done):
    return 1 if np.max(new) >= 10 else 0

def rewardMaxTile(new, old, done):
    r = .01 if np.max(new) >= 11 else 0
    return r

def rewardMaxTileSplits(new, old, done):
    r = .1024 if np.max(new)    == 12 else \
        .0064 if np.max(new)   == 11 else \
        .0004 if np.max(new)  == 10 else \
        .0001 if np.max(new)  == 9 else 0
    return r

def manualreward(new, old):
    reward = 0
    nonmoves = 0
    for r in range(4):
        for c in range(4):
            if(new[r,c] == old[r,c]):
                nonmoves += 1
            reward += (new[r,c])*weights[r,c] # 0 if new[r,c] == 0 else math.log(new[r,c],2)
    return reward  if nonmoves < 16 else 0 #- 10*unsortedness(a)

def mergeMaxReward(new, old):
    return 0 if np.equal(new, old).all() else ((merges(new, old)-5)/10 + np.max(new)*.1)

def mergeNewMaxReward(new, old): #more spread out large rewards
    newmax = np.max(new)
    r = np.max(new) if np.max(new) > np.max(old) else 0
    r += (merges(new, old)-5)/10
    return 0 if np.equal(new, old).all() else r

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

def bestNewStates(a):
    u = np.copy(a.T)
    d = np.flip(np.copy(a.T), axis = 1)
    r = np.flip(np.copy(a), axis = 1)
    l = np.copy(a)

    u = moveLeftAndCombine(u).T
    d = np.flip(moveLeftAndCombine(d), axis = 1).T
    r = np.flip(moveLeftAndCombine(r), axis = 1)
    l = moveLeftAndCombine(l)

    uu = [np.copy(u)] if boardEquals(a, u) else addBestSpawn(np.copy(u))
    dd = [np.copy(d)] if boardEquals(a, d) else addBestSpawn(np.copy(d))
    rr = [np.copy(r)] if boardEquals(a, r) else addBestSpawn(np.copy(r))
    ll = [np.copy(l)] if boardEquals(a, l) else addBestSpawn(np.copy(l))

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

def addBestSpawn(a):
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
            if(r <= maxr):
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

"""# Network structure"""
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
        p = F.relu(p)
        #Size changes from (20, 32, 32) to (30, 32, 32)
        p = self.convP2(p)
        #print('conv2 shape: ' + str(x.shape))
        p = self.convP2_drop(p)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        p = F.max_pool2d(p, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        p = F.relu(p)
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
        v = F.relu(v)
        #Size changes from (20, 32, 32) to (30, 32, 32)
        v = self.convV2(v)
        #print('conv2 shape: ' + str(x.shape))
        v = self.convV2_drop(v)
        #print('drop  shape: ' + str(x.shape))
        #Size changes from (30, 32, 32) to (30, 16, 16)
        v = F.max_pool2d(v, kernel_size=2)
        #print('pool2 shape: ' + str(x.shape))
        v = F.relu(v)
        v = v.view((-1, 1024))
        #print('post view shape: ' + str(x.shape))
        v = self.fcV1(v)


        return p, v

    def loss_func(self, states, actions, rewards, last_state_value):
        self.train()

        logits, values = self.forward(states)

        valuesnp = values.data.numpy().reshape(len(rewards))

        lam = 1
        delta_t = np.array(rewards) + gamma * np.append(valuesnp[1:], last_state_value) - valuesnp
        advantage = []
        prev_val = 0
        for d in delta_t[::-1]:
            prev_val = d + lam * gamma * prev_val
            advantage.append(prev_val)
        advantage.reverse()
        advantage = torch.FloatTensor(advantage)

        discounted_rewards = []
        for r in rewards[::-1]:    # discount rewards adjusted by next_state's value
            last_state_value = r + gamma * last_state_value
            discounted_rewards.append(last_state_value)
        discounted_rewards.reverse()
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        values_difference = discounted_rewards - values
        c_loss = values_difference ** 2

        probs = F.softmax(logits, dim=1)
        '''probs = F.relu(probs)
        probs = probs + 1e-10'''
        m = self.distribution(probs)
        prob_entropy = m.entropy()*0



        a_loss = -1 * m.log_prob(actions) * advantage.detach().squeeze()

        c_loss *= .05
        a_loss *= .1
        #a_loss = prob_entropy + action_advantage
        print('weighted critic loss: {}, weighted actor loss: {}, weighted probability entropy :{}'.format(c_loss.mean(), a_loss.mean(), prob_entropy.mean()))
        return (c_loss + a_loss + prob_entropy).mean()

"""# A3C"""

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Runner(mp.Process):
    def __init__(self, brain, optimizer, global_episode, global_episode_reward, res_queue, name, maxTileAvg, maxTileQueue, lossQueue, lossAvg, largestTileAvgSeen):
        super(Runner, self).__init__()
        self.name = 'Runner_%02i' % name
        self.global_episode, self.global_episode_reward, self.res_queue = global_episode, global_episode_reward, res_queue
        self.global_net, self.optimizer = brain, optimizer
        self.local_net = load_model('2048net.th')           # local network
        self.agent = myAgent()
        self.maxTileAvg = maxTileAvg
        self.maxTileQueue = maxTileQueue
        self.lossAvg = lossAvg
        self.lossQueue = lossQueue
        self.largestTileAvgSeen = largestTileAvgSeen

    def run(self):
        total_step = 1
        while self.global_episode.value < max_eps:
            #print("global episode value: {} for worker {}".format(self.global_episode.value, self.name))
            s = self.agent.reset()
            statebuffer, actionbuffer, rewardbuffer = [], [], []
            ep_reward = 0.
            while True:
                s = power2mat3d(s)
                self.local_net.eval()

                logits, _ = self.local_net.forward(torch.FloatTensor(s.reshape((1,16,4,4))))
                logits[logits != logits] = 0 #removes Nans
                probs = F.softmax(logits, dim=1).data
                #print(probs)
                a = self.agent.selectChoiceAction(probs)

                next_state, reward, done, maxTile = self.agent.step(a)
                '''if done: reward -= 16 if np.max(next_state) == 6 else \
                                    8 if np.max(next_state) == 7 else \
                                    4 if np.max(next_state) == 8 else \
                                    2 if np.max(next_state) == 9 else 0'''
                ep_reward += reward
                actionbuffer.append(a)
                statebuffer.append(s)
                rewardbuffer.append(reward)

                if total_step % update_global == 0 or done:  # update global and assign to local net
                    # sync
                    #print("state we are getting value for: {}".format(s_))
                    update(self.optimizer, self.local_net, self.global_net, done, power2mat3d(next_state), statebuffer, actionbuffer, rewardbuffer, self.lossAvg, self.lossQueue, gamma)
                    statebuffer, actionbuffer, rewardbuffer = [], [], []

                    if done:  # done and print information
                        print('{} finished w/ reward {:3.4f}\t and max tile {:2.0f}'.format(self.name, round(ep_reward, 3), maxTile), end = ' @\t')
                        store_episode(self.global_episode, self.global_episode_reward, ep_reward, self.res_queue, self.name, maxTile, self.maxTileAvg, self.maxTileQueue, self.local_net, self.largestTileAvgSeen)
                        self.lossQueue.put(None)
                        break
                s = next_state
                total_step += 1
        self.res_queue.put(None)
        self.lossQueue.put(None)

"""# A3C helpers"""

def update(optimizer, local_net, brain, done, next_state, batch_states, batch_actions, batch_rewards, lossAvg, lossQueue, gamma = .99):
    if done:
        last_state_value = 0.               # terminal
    else:
        last_state_value = local_net.forward(torch.FloatTensor(next_state.reshape((1,16,4,4))))[-1].data.numpy()[0, 0]
    #print("value of last state in update: {}".format(value_s_))

    #global advantage estimation

    optimizer.zero_grad()

    lossacc = []
    #loss_sum = 0
    '''for i in range(len(batch_actions)): #accumulated loss
        loss = local_net.loss_func(
            torch.FloatTensor([batch_states[i]]),
            torch.FloatTensor([batch_actions[i]]),
            torch.FloatTensor([buffer_value_target[i]]))/len(batch_actions)
        #loss_sum += loss
        loss.backward()
        lossacc.append(loss.item())
    #loss_sum.mean().backward()'''

    loss = local_net.loss_func(
        torch.FloatTensor(batch_states),
        torch.FloatTensor(batch_actions),
        batch_rewards,
        last_state_value)
    lossacc.append(loss.item())
    loss.backward()

    with lossAvg.get_lock(): #running average for plotting computation
        if lossAvg.value == 0.:
            lossAvg.value = sum(lossacc)/len(lossacc)
        else:
            lossAvg.value = lossAvg.value * gamma + sum(lossacc)/len(lossacc) * (1-gamma)
    lossQueue.put(lossAvg.value)

    torch.nn.utils.clip_grad_norm_(local_net.parameters(), 3) #gradient clipping
    #update the global brain
    for lp, gp in zip(local_net.parameters(), brain.parameters()):
        gp._grad = lp.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(brain.state_dict())


def store_episode(global_episode, global_episode_r, episode_reward, res_queue, name, maxTile, maxTileAvg, maxTileQueue, local_net, largestTileAvgSeen):
    with global_episode.get_lock():
        global_episode.value += 1
    with global_episode_r.get_lock():
            global_episode_r.value = global_episode_r.value * gamma + episode_reward * (1-gamma)
    with maxTileAvg.get_lock():
        if maxTileAvg.value == 0.:
            maxTileAvg.value = .9 * maxTile
        else:
            maxTileAvg.value = maxTileAvg.value * gamma + maxTile * (1-gamma)
    with largestTileAvgSeen.get_lock():
        if maxTileAvg.value > largestTileAvgSeen.value:
            largestTileAvgSeen.value =  maxTileAvg.value
            save_model(local_net, '4brain'+str(update_global)+'.th')
            print('saving new best model', end = ' ')
    maxTileQueue.put(maxTileAvg.value)
    res_queue.put(global_episode_r.value)

    print(name,"episode:", global_episode.value," running average reward: {:3.2f} and running max tile {:2.3f}".format(round(global_episode_r.value, 3), round(maxTileAvg.value,3)))


if __name__ == "__main__":
    brain = load_model('2048net.th')        # global network
    brain.share_memory()         # share the global parameters in multiprocessing
    optimizer = SharedAdam(brain.parameters(), lr=1e-4)      # global optimizer betas=(0.92, 0.999)
    maxTileQueue, maxTileAvg = mp.Queue(), mp.Value('d', 8.)
    largestTileAvgSeen = mp.Value('d', 8.)
    lossQueue, lossAvg = mp.Queue(), mp.Value('d', 0.)
    global_episode, global_episode_reward = mp.Value('i', 0), mp.Value('d', 0.)
    rewards_queue = mp.Queue()

    # parallel training
    runners = [Runner(brain, optimizer, global_episode, global_episode_reward, rewards_queue, i, maxTileAvg, maxTileQueue, lossQueue, lossAvg, largestTileAvgSeen) for i in range(mp.cpu_count())] #mp.cpu_count()
    try:
        [a.start() for a in runners]
        print("finished creating & starting Runners, entering main loop")
        results = []
        maxTileList = []                 # record episode reward to plot
        losslist = []
        while True:
            m = maxTileQueue.get()
            r = rewards_queue.get()
            l = lossQueue.get()
            if r is not None:
                maxTileList.append(m)
                results.append(r)
                while l is not None:
                    losslist.append(l)
                    l = lossQueue.get()
            else:
                break

    finally:
        print("wait for all runners to finish")
        [a.join() for a in runners]

        fig, ax1 = plt.subplots(1,2)
        color = 'tab:green'
        ax1[0].set_xlabel('episodes')
        ax1[0].set_ylabel('100 episode reward running average', color=color)
        ax1[0].plot(results, color=color)
        ax1[0].tick_params(axis='y', labelcolor=color)

        ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('100 episode max tile running average', color=color)  # we already handled the x-label with ax1
        ax2.plot(maxTileList, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax1[1].plot(losslist)

        plt.show()
