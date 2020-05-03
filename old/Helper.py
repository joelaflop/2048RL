import numpy as np
import random
import math

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

    b[r, c] = 2 if random.random() < .9 else 4
    b[rr, cc] = 2 if random.random() < .9 else 4

    return b

weights = np.array([[2**2,2,1,0],
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
            a2[spawn] = 2
            a4 = np.copy(a)
            a4[spawn] = 4
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

def zigzagOrder(a):
    return np.concatenate((np.flip(a[0]), a[1], np.flip(a[2]), a[3]))
