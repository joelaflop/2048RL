from WebDriver import myWebDriver
import numpy as np

class myAgent():
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
