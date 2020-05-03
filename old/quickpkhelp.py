from Agent import myAgent
import random
import numpy as np
from Logger import myLogger
import BoardTree2 as b
import pandas as pd

episodes = 100
moves = 10000

store = False
log = False

counter = 0
donecount = 0
totTile = 0
totTiles = np.zeros(16)

train = np.empty((episodes*moves,17))
valid = np.empty((episodes*moves,17))
merges = 0

for i in range (episodes):
    print("round: "+str(i))
    agent = myAgent()
    if log:
        logger = myLogger(name = 'Looks'+str(i))
    board = agent.reset()
    for t in range(moves):
        head = b.BoardTree(board, num_levels = 5)

        rewards = head.calc_rewards()
        action = np.argmax(np.array(rewards)) #action = random.randint(0,3)

        if log:
            logger.log(board, merges, action)
        if store:
            row = np.append(board.flatten(), action).reshape((1,17))
            if(random.random() < .8):
                train[counter, :] = row
            else:
                valid[counter, :] = row
            counter += 1

        board, _, done, merges = agent.step(action)

        if done:
            print("Episode finished after {} moves".format(t+1))
            bestTile = int(np.max(board))
            totTiles[bestTile-1] += 1
            donecount += 1
            break
print("tile distr: "+str(totTiles))
print("avg tile: "+str(np.dot(totTiles, np.arange(1,17))/np.sum(totTiles)))
agent.close()
logger.close()
if store:
    tdf = pd.DataFrame(train[~np.all(train == 0, axis=1)])
    vdf = pd.DataFrame(valid[~np.all(valid == 0, axis=1)])
    tdf.to_csv('train_lookaheaddata.csv', index = False)
    vdf.to_csv('valid_lookaheaddata.csv', index = False)

print("average best tile: "+str(totTile/(episodes)))
print((donecount == episodes))
