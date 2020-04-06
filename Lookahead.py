from Agent import webAgent
import random
import numpy as np
from Logger import myLogger
from BoardTree import BoardTree



'''
https://gym.openai.com/docs/
'''

try:
    agent = webAgent()
    logger = myLogger(name ='RL')
    board = agent.reset()
    for t in range(20000):

        head = BoardTree(board, num_levels = 5)

        rewards = head.calc_rewards()
        action = np.argmax(np.array(rewards))
        logger.logold(head, board, rewards, action)

        board, score, done, info = agent.step(action)
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
