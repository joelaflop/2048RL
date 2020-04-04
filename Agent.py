from WebDriver import myWebDriver
import numpy as np

class myAgent():
    def __init__ (self):
        self.driver = myWebDriver()

    def reset(self):
        return self.driver.getBoard()

    def step(self, d):
        oldScore, newScore = self.driver.move(d)
        reward = newScore - oldScore

        done = False

        info = 0
        print('-------')
        observation = self.driver.getBoard()


        return observation, reward, done, info

    def close(self):
        self.driver.close()
