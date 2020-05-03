from WebDriver import myWebDriver
import numpy as np
import HelperBase2 as h
import random


class myAgent():
    def __init__ (self):
        print('agent open')

    def reset(self):
        self.observation = h.newBoard()
        self.steps = 0
        return self.observation

    def step(self, d):
        qstates, nstates = h.randomNewStates(self.observation)
        newBoard = random.choice(nstates[d])

        done = (h.boardEquals(newBoard, self.observation))

        reward = h.mergeMaxReward(newBoard, self.observation)

        info = h.merges(newBoard, self.observation)

        self.observation = newBoard

        self.steps += 1
        return self.observation, reward, done, info

    def close(self):
        print('agent closed')

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
