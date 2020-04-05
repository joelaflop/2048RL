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
    for t in range(1000):
        Qup, Qdown, Qright, Qleft, Sup, Sdown, Sright, Sleft = helper.newStates(observation)

        Rs = np.empty(4)

        Rs[0] = helper.reward(Qup, observation)
        Rs[1] = helper.reward(Qdown, observation)
        Rs[2] = helper.reward(Qright, observation)
        Rs[3] = helper.reward(Qleft, observation)

        action = np.argmax(Rs)

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


'''import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()'''
