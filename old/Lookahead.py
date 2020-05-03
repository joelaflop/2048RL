from Agent import myAgent
import random
import numpy as np
from Logger import myLogger
import BoardTree2 as b
import pandas as pd
import matplotlib.pyplot as plt

episodes = 300
maxmoves = 10000

store = True
log = True

donecount = 0
totTile = 0
totTiles = np.zeros(16)

gamma = .99

merges = 0

agent = myAgent()

episodearrays = []

data = {'duration' : [], 'total reward' : [], 'color' : []}

for i in range (episodes):
    print("round: "+str(i))
    totalvalreward = 0
    valuerewards = []
    counter = 0

    if log:
        logger = myLogger(name = 'Looks'+str(i))
    if store:
        currarr = np.zeros((maxmoves,17))
    board = agent.reset()
    for t in range(maxmoves):
        head = b.BoardTree(board, num_levels = 5)

        rewards = head.calc_rewards()
        action = np.argmax(np.array(rewards)) #action = random.randint(0,3)

        newboard, valuereward, done, merges = agent.step(action)
        valuerewards.append(valuereward)
        totalvalreward += valuereward

        if log:
            logger.log(board, merges, action)
        if store:
            row = np.append(board.flatten(), action).reshape((1,17))
            currarr[counter, :] = row
            counter += 1
        board = newboard
        if done:
            print("Episode finished after {} moves".format(t+1))
            bestTile = int(np.max(board))
            print('best tile: {}'.format(bestTile))
            print('total value reward: {}'.format(totalvalreward))
            totTiles[bestTile-1] += 1
            donecount += 1
            if store:
                r = np.array(valuerewards)[::-1].cumsum()[::-1].reshape((-1,1))
                currarr = currarr[~np.all(currarr == 0, axis=1)]
                currarr = np.append(currarr, r, axis = 1)
                episodearrays.append(np.copy(currarr))

            data['duration'].append(t)
            data['total reward'].append(totalvalreward)
            data['color'].append(0)
            break
if donecount == episodes:
    print("tile distr: "+str(totTiles))
    print("avg best tile: "+str(np.dot(totTiles, np.arange(1,17))/np.sum(totTiles)))
    plt.scatter('duration', 'total reward', c = 'color', data=data)
    plt.xlabel('duration')
    plt.ylabel('total reward')
    plt.show()
else:
    print("number of completions didn't match number of episodes")
agent.close()
if log:
    logger.close()
if store:
    df = pd.DataFrame(np.concatenate(episodearrays, axis = 0))
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    train.to_csv('train_lookaheaddata.csv', index = False)
    test.to_csv('valid_lookaheaddata.csv', index = False)
