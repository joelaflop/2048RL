from Agent import myAgent
import random
import numpy as np
from Logger import myLogger
import Helper as h
import BoardTree2 as b
import pandas as pd

trials = 100
moves = 10000

counter = 0

train = np.empty((trials*moves,17))
valid = np.empty((trials*moves,17))

for i in range (trials):
    print("round: "+str(i))
    try:
        agent = myAgent()
        logger = myLogger(name = 'Looks'+str(i))
        board = agent.reset()
        for t in range(moves):

            #action = random.randint(0,3)

            head = b.BoardTree(board, num_levels = 5)

            rewards = head.calc_rewards()
            action = np.argmax(np.array(rewards))

            logger.log(board, 123457654, action)

            row = np.append(board.flatten(), action).reshape((1,17))
            if(random.random() < .8):
                train[counter, :] = row
            else:
                valid[counter, :] = row
            counter += 1

            board, _, done, _ = agent.step(action)



            if done:
                print("Episode finished after {} moves".format(t+1))
                break
        pass
    except Exception as e:
        e.trace
    finally:
        agent.close()
        logger.close()
        pass
tdf = pd.DataFrame(train[~np.all(train == 0, axis=1)])
vdf = pd.DataFrame(valid[~np.all(valid == 0, axis=1)])
tdf.to_csv('train_lookaheaddata.csv', index = False)
vdf.to_csv('valid_lookaheaddata.csv', index = False)
