from Agent import myAgent
import random
import numpy as np
from Helper2048 import Board
from Logger import myLogger


'''
https://gym.openai.com/docs/
'''

try:
    agent = myAgent()
    logger = myLogger()
    observation = agent.reset()
    for t in range(20000):

        head = Board(observation, num_levels = 7)

        rewards = head.calc_rewards()
        action = np.argmax(np.array(rewards))
        logger.log(head, observation, rewards, action)

        observation, score, done, info = agent.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    pass
except Exception as e:
    print(e)
finally:
    agent.close()
    logger.close()
    pass
