import numpy as np
import random

class Board():
    def __init__ (self, board, r = 0, p = 1, move = -1, num_levels = 2):
        self.qstates = qStates(board)
        self.nstates = newStates(self.qstates) #4 lists of lists
        self.move = move
        self.p = p
        self.reward = r

        count = 0
        for l in self.nstates:
            count += len(l)

        self.children = []

        if num_levels > 0:
            for dir in self.nstates:
                l = []
                for b in dir:
                    l.append(Board(b, r = reward(b, board), p = self.p/count, move = dir, num_levels = num_levels-1))
                self.children.append(l)

    def calc_rewards(self):
        rewards = []
        for dir in self.children:
            reward = 0
            for child in dir:
                reward = max(reward, child.child_rewards())
            rewards.append(reward)
        return rewards

    def child_rewards(self):
        if(len(self.children) > 0):
            for dir in self.children:
                for child in dir:
                    return max(self.reward, child.child_rewards())
        else:
            return self.reward


'''
helper functions
'''
weights = np.array([[2**0,2**1,2**2,2**3],
                    [2**4,2**5,2**6,2**7],
                    [2**11,2**10,2**9,2**8],
                    [2**12,2**13,2**14,2**15]])

def reward(a, old):
    reward = 0
    nonmoves = 0
    for r in range(4):
        for c in range(4):
            reward += (a[r,c])*weights[r,c]

    return reward

def qStates(a):
    u = np.copy(a.T)
    d = np.flip(np.copy(a.T), axis = 1)
    r = np.flip(np.copy(a), axis = 1)
    l = np.copy(a)

    u = moveLeftAndCombine(u).T
    d = np.flip(moveLeftAndCombine(d), axis = 1).T
    r = np.flip(moveLeftAndCombine(r), axis = 1)
    l = moveLeftAndCombine(l)

    return [u,d,r,l]

def newStates(l):
    uu = addWorstSpawn(np.copy(l[0]))
    dd = addWorstSpawn(np.copy(l[1]))
    rr = addWorstSpawn(np.copy(l[2]))
    ll = addWorstSpawn(np.copy(l[3]))

    return [uu, dd, rr, ll]

def addRandomSpawns(a):
    zlist = []
    for r in range(4):
        for c in range(4):
            if a[r, c] == 0:
                zlist.append((r,c))
    if zlist:
        alist = []
        for spawn in zlist:
            aa = np.copy(a)
            aa[spawn] = 2
            alist.append(aa)
            #aa[spawn] = 2 if random.random() < .9 else 4
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
            aa = np.copy(a)
            if(r >= maxr):
                if(c >= maxc):
                    maxr = r
                    maxc = c
                    aa[(maxr, maxc)] = 2
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
                a[r, c] *= 2
                a[r, c+1] = 0
    return a
