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
update_global = 10
max_eps = 2000000

os.environ["OMP_NUM_THREADS"] = "1"

"""# Data helpers"""
def save_model(model, name):
    if isinstance(model, Net):
        return save(model.state_dict(), path.join(path.abspath(''), name))
    #return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))

'''
helper functions
'''



"""# Network structure"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Input channels = 1, output channels = 20
        self.fcP1 = torch.nn.Linear(2, 128)
        self.fcP2 = torch.nn.Linear(128, 256)
        self.fcP3 = torch.nn.Linear(256, 3)

        self.fcV1 = torch.nn.Linear(2, 128)
        self.fcV2 = torch.nn.Linear(128, 256)
        self.fcV3 = torch.nn.Linear(256, 1)

        self.distribution = torch.distributions.Categorical
        self.epsilon = 1e-7

    def forward(self, x):
        p = self.fcP1(x)
        p = F.relu(p)
        p = self.fcP2(p)
        p = F.relu(p)
        p = self.fcP3(p)


        v = self.fcV1(x)
        v = F.relu(v)
        v = self.fcV2(v)
        v = F.relu(v)
        v = self.fcV3(v)

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

        probs = F.softmax(logits, dim=2)
        '''probs = F.relu(probs)
        probs = probs + 1e-10'''
        m = self.distribution(probs)
        prob_entropy = m.entropy()*1e-2

        c_loss *= .5

        a_loss = -1 * m.log_prob(actions) * advantage.detach().squeeze()
        #a_loss = prob_entropy + action_advantage
        #print('weighted critic loss: {}, weighted actor loss: {}, weighted probability entropy :{}'.format(c_loss.mean(), a_loss.mean(), prob_entropy.mean()))
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

import gym
class Runner(mp.Process):
    def __init__(self, brain, optimizer, global_episode, global_episode_reward, res_queue, name, maxTileAvg, maxTileQueue, lossQueue, lossAvg, largestTileAvgSeen):
        super(Runner, self).__init__()
        self.name = 'Runner_%02i' % name
        self.global_episode, self.global_episode_reward, self.res_queue = global_episode, global_episode_reward, res_queue
        self.global_net, self.optimizer = brain, optimizer
        self.local_net = Net()          # local network
        self.agent = gym.make('MountainCar-v0')
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
                self.local_net.eval()

                logits, _ = self.local_net.forward(torch.FloatTensor(s.reshape((1,1,2))))
                #print('logits: {}'.format(logits))
                #logits[logits != logits] = 0 #removes Nans
                probs = F.softmax(logits, dim=2)
                #print(probs)

                m = torch.distributions.Categorical(probs)
                a = m.sample().data.numpy()[0,0]

                next_state, reward, done, _ = self.agent.step(a)
                '''reward = next_state[0] + .5
                if next_state[0] >= 0.5:
                    reward += 1'''
                ep_reward += reward
                actionbuffer.append(a)
                statebuffer.append(s.reshape((1,2)))
                rewardbuffer.append(reward)

                if(self.name == 'Runner_00'):
                    self.agent.render()

                if total_step % update_global == 0 or done:  # update global and assign to local net
                    # sync
                    #print("state we are getting value for: {}".format(s_))
                    update(self.optimizer, self.local_net, self.global_net, done, next_state.reshape((2,1,1)), statebuffer, actionbuffer, rewardbuffer, self.lossAvg, self.lossQueue, gamma)
                    statebuffer, actionbuffer, rewardbuffer = [], [], []

                    if done:  # done and print information
                        print('{} finished w/ reward {:3.4f}\t'.format(self.name, round(ep_reward, 3)), end = ' @\t')
                        store_episode(self.global_episode, self.global_episode_reward, ep_reward, self.res_queue, self.name, self.local_net)
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
        last_state_value = local_net.forward(torch.FloatTensor(next_state.reshape((1,1,2))))[-1].data.numpy()[0, 0]
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


def store_episode(global_episode, global_episode_r, episode_reward, res_queue, name, local_net):
    with global_episode.get_lock():
        global_episode.value += 1
    with global_episode_r.get_lock():
            global_episode_r.value = global_episode_r.value * gamma + episode_reward * (1-gamma)
    res_queue.put(global_episode_r.value)

    print(name,"episode:", global_episode.value," running average reward: {:3.2f}".format(round(global_episode_r.value, 3)))


if __name__ == "__main__":
    brain = Net()      # global network
    brain.share_memory()         # share the global parameters in multiprocessing
    optimizer = SharedAdam(brain.parameters(), lr=1e-4)      # global optimizer betas=(0.92, 0.999)
    maxTileQueue, maxTileAvg = mp.Queue(), mp.Value('d', 9.)
    largestTileAvgSeen = mp.Value('d', 9.)
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
            r = rewards_queue.get()
            l = lossQueue.get()
            if r is not None:
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
