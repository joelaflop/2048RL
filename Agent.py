from WebDriver import myWebDriver
import numpy as np
import Helper2048 as h
import random


class myAgent():
    def __init__ (self):
        print('open')

    def reset(self):
        self.observation = h.newBoard()
        return self.observation

    def step(self, d):
        qstates, nstates = h.randomNewStates(self.observation)
        newBoard = random.choice(nstates[d])

        done = ((self.observation.astype(int) == qstates[0].astype(int)).all() and
                (self.observation.astype(int) == qstates[1].astype(int)).all() and
                (self.observation.astype(int) == qstates[2].astype(int)).all() and
                (self.observation.astype(int) == qstates[3].astype(int)).all())

        reward = h.manualreward(newBoard, self.observation)
        self.observation = newBoard

        info = 'no info'


        return self.observation, reward, done, info

    def close(self):
        print('done')

class webAgent():
    def __init__ (self):
        self.driver = myWebDriver()


    def reset(self):
        self.observation = self.driver.getBoard()
        return self.observation

    def step(self, d):
        oldScore, newScore = self.driver.move(d)

        done = False

        info = self.driver.getInfo()

        self.observation = self.driver.getBoard()

        return self.observation, newScore, done, info

    def close(self):
        self.driver.close()
