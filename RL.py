from Agent import myAgent
import random
'''
https://gym.openai.com/docs/
'''

try:
    agent = myAgent()
    observation = agent.reset()
    print(observation)
    for t in range(2):
        #env.render()
        action = random.randint(0,3)
        observation, reward, done, info = agent.step(action)
        print(observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    pass
except Exception as e:
    print (e)
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
