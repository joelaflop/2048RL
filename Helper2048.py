import numpy as np
import random



class Board():
    def __init__ (self, board, r = 0, p = 1, move = -1, num_levels = 2):
        self.move = move
        self.p = p
        self.reward = r
        self.board = board
        self.children = []

        if num_levels > 0:
            qstates, nstates = newStates(self.board) #4 lists of lists
            count = 0
            for l in nstates:
                count += len(l)
            for d, dir in enumerate(nstates):
                l = []
                for b in dir:
                    if not boardEquals(b, self.board):
                        l.append(Board(b, r = reward(b, self.board), p = self.p/count, move = d, num_levels = num_levels-1))
                self.children.append(l)

    def calc_rewards(self):
        rewards = []
        for dir in self.children:
            r = 0
            for child in dir:
                r = max(r, child.child_rewards())
            rewards.append(r)
        return rewards

    def child_rewards(self):
        if(len(self.children) > 0):
            r = 0
            for dir in self.children:
                for child in dir:
                    r = max(r, self.reward*self.p + child.child_rewards())
            return r if self.reward > 0 else 0
        else:
            return self.reward*self.p


'''
helper functions
'''
weights = np.array([[0,2**0,2**1,2**2],
                    [2**3,2**4,2**5,2**6],
                    [2**10,2**9,2**8,2**7],
                    [2**16,2**17,2**18,2**19]])

def reward(a, old):
    reward = 0
    nonmoves = 0
    for r in range(4):
        for c in range(4):
            if(a[r,c] == old[r,c]):
                nonmoves += 1
            reward += (a[r,c])*weights[r,c]
    return reward if nonmoves < 16 else 0


def newStates(a):
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
            #aa[spawn] = 2 if random.random() < .9 else 4
            alist.append(aa)
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

    valid_mask = a != 0
    flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(a.shape[1]-1,-1,-1)
    flipped_mask = flipped_mask[:,::-1]
    a[flipped_mask] = a[valid_mask]
    a[~flipped_mask] = 0

    return a

def boardEquals(a, b):
    e = 0
    for r in range(4):
        for c in range(4):
            if(a[r,c] == b[r,c]):
                e += 1
    return e == 16
