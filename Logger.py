import numpy as np

class myLogger():
    def __init__(self):
        self.f = open('log', 'w')

    def log(self, head, observation, rewards, action):
        self.f.write(str(observation))
        self.f.write('\n')
        self.f.write(str(rewards))
        self.f.write(' ')
        self.f.write(str(action))
        self.f.write('\n')
        self.f.write(str(head.children[action][0].board))
        self.f.write('\n')
        self.f.write('\n')
        self.f.write('\n')


    def write(self, s):
        self.f.write(s+' ')


    def close(self):
        self.f.close()

f = myLogger()
