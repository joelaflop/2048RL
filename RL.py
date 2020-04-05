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
        '''Qstates1 = helper.qStates(observation)
        Nstates1 = helper.newStates(Qstates1) #now list of lists of boards

        rewards1 = []
        Qstates2 = []
        for l in Nstates1:
            rewardsl = helper.reward(l, observation)
            rewards1.append(sum(rewardsl) / len(rewardsl))
            subQ = []
            for b in l:
                subQ.append(helper.qStates(b))
            Qstates2.append(subQ)'''

        head = helper.Board(observation, move = 5, num_levels = 5)

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
        print(helper.reward(head.qstates[action], observation))
        print('down')
        print(helper.reward(head.qstates[1], observation))
        print('right')
        print(helper.reward(head.qstates[2], observation))
        print('-')


        observation, reward, done, info = agent.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break





        '''
        Qstates2 = [helper.qStates(Nstates1[0]),
                    helper.qStates(Nstates1[1]),
                    helper.qStates(Nstates1[2]),
                    helper.qStates(Nstates1[3])]

        Nstates2 = [helper.newStates(Qstates2[0]),
                    helper.newStates(Qstates2[1]),
                    helper.newStates(Qstates2[2]),
                    helper.newStates(Qstates2[3])]

        #rewards1 = helper.reward(Nstates1, observation)
        rewards2 = [[new*rewards1[0] for new in helper.reward(Nstates2[0], Nstates1[0])],
                    [new*rewards1[1] for new in helper.reward(Nstates2[1], Nstates1[1])],
                    [new*rewards1[2] for new in helper.reward(Nstates2[2], Nstates1[2])],
                    [new*rewards1[3] for new in helper.reward(Nstates2[3], Nstates1[3])]]
        print(np.array(rewards2))'''




        #action = int(np.argmax(np.array(rewards2))/4)
        '''action = np.argmax(np.array(rewards1))

        print(action)
        '''
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
