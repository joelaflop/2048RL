import numpy as np
import random

def reward(a, old):
    reward = 0
    nonmoves = 0
    for r in range(4):
        for c in range(4):
            if a[r,c] == old[r,c]:
                nonmoves += 1
            reward += (a[r,c])*(r+2)*(c)
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

    uu = addRandomSpawn(np.copy(u))
    dd = addRandomSpawn(np.copy(d))
    rr = addRandomSpawn(np.copy(r))
    ll = addRandomSpawn(np.copy(l))

    return u, d, r, l, uu, dd, rr, ll

def addRandomSpawn(a):
    zlist = []
    for r in range(4):
        for c in range(4):
            if a[r, c] == 0:
                zlist.append((r,c))
    if zlist:
        spawn = random.choice(zlist)
        a[spawn] = 2 if random.random() < .9 else 4
    return a

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
