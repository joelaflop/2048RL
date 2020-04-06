from Agent import webAgent, myAgent
import random
import numpy as np
from Logger import myLogger
from BoardTree import BoardTree
import Helper2048 as h


'''
https://gym.openai.com/docs/
'''
for i in range (50):
    try:
        agent = myAgent()
        logger = myLogger(name = 'NN.'+str(i))
        board = agent.reset()
        for t in range(50000):

            #action = random.randint(0,3)

            head = BoardTree(board, num_levels = 5)

            rewards = head.calc_rewards()
            action = np.argmax(np.array(rewards))

            board, reward, done, info = agent.step(action)

            logger.log(board, reward, action)
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
