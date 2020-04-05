from Agent import myAgent
import random
import numpy as np
import Helper2048 as helper
'''
https://gym.openai.com/docs/
'''

try:
    agent = myAgent()
    observation = agent.reset()
    for t in range(10000):

        head = helper.Board(observation, num_levels = 7)

        rewards = head.calc_rewards()

        args = np.argsort(np.array(rewards))
        i = 3
        action = args[i]
        while (np.equal(observation, head.qstates[action]).all()):
            i -=1
            action = args[i]

        print(observation)
        print(rewards)
        print(action)


        observation, reward, done, info = agent.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    pass
except Exception as e:
    print (e.trace)
finally:
    agent.close()
    pass
